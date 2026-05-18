
/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ppocr_system.h"
#include "image_utils.h"
#include "image_drawing.h"
#include "file_utils.h"
#include <opencv2/opencv.hpp>

#define INDENT "    "
#define THRESHOLD 0.3                                // pixel score threshold
#define BOX_THRESHOLD 0.6                            // box score threshold
#define USE_DILATION false                           // whether to do dilation, true or false
#define DB_SCORE_MODE "slow"                         // slow or fast. slow for polygon mask; fast for rectangle mask
#define DB_BOX_TYPE "poly"                           // poly or quad. poly for returning polygon box; quad for returning rectangle box
#define DB_UNCLIP_RATIO 1.5                          // unclip ratio for poly type


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("%s <det_model_path> <rec_model_path> [camera_index]\n", argv[0]);
        return -1;
    }

    const char* det_model_path = argv[1];
    const char* rec_model_path = argv[2];
    int camera_index = 44;
    camera_index = atoi(argv[3]);

    int ret;
    ppocr_system_app_context rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(ppocr_system_app_context));

    ret = init_ppocr_model(det_model_path, &rknn_app_ctx.det_context);
    if (ret != 0) {
        printf("init_ppocr_model fail! ret=%d det_model_path=%s\n", ret, det_model_path);
        return -1;
    }

    ret = init_ppocr_model(rec_model_path, &rknn_app_ctx.rec_context);
    if (ret != 0) {
        printf("init_ppocr_model fail! ret=%d rec_model_path=%s\n", ret, rec_model_path);
        return -1;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ppocr_text_recog_array_result_t results;
    ppocr_det_postprocess_params params;
    params.threshold = THRESHOLD;
    params.box_threshold = BOX_THRESHOLD;
    params.use_dilate = USE_DILATION;
    params.db_score_mode = DB_SCORE_MODE;
    params.db_box_type = DB_BOX_TYPE;
    params.db_unclip_ratio = DB_UNCLIP_RATIO;
    const unsigned char blue[] = {0, 0, 255};

    cv::VideoCapture cap(camera_index);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open the camera." << std::endl;
        return -1;
    }

    cv::Mat src_frame;
    cv::namedWindow("out", cv::WINDOW_NORMAL);
    cv::setWindowProperty("out", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    while(true){
        cap >> src_frame;
        if (src_frame.empty()) {
            std::cerr << "Failed to grab frame from the camera." << std::endl;
            break;
        }

        cv::rotate(src_frame, src_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
        src_image.height = src_frame.rows;
        src_image.width = src_frame.cols;
        src_image.width_stride = src_frame.step[0];
        src_image.virt_addr = src_frame.data;
        src_image.format = IMAGE_FORMAT_RGB888;
        src_image.size = src_frame.total() * src_frame.elemSize();

        ret = inference_ppocr_system_model(&rknn_app_ctx, &src_image, &params, &results);

        // Draw Objects
        for (int i = 0; i < results.count; i++)
        {
            printf("[%d] @ [(%d, %d), (%d, %d), (%d, %d), (%d, %d)]\n", i,
                results.text_result[i].box.left_top.x, results.text_result[i].box.left_top.y, results.text_result[i].box.right_top.x, results.text_result[i].box.right_top.y, 
                results.text_result[i].box.right_bottom.x, results.text_result[i].box.right_bottom.y, results.text_result[i].box.left_bottom.x, results.text_result[i].box.left_bottom.y);
            //draw Quadrangle box
            draw_line(&src_image, results.text_result[i].box.left_top.x, results.text_result[i].box.left_top.y, results.text_result[i].box.right_top.x, results.text_result[i].box.right_top.y, 255, 2);
            draw_line(&src_image, results.text_result[i].box.right_top.x, results.text_result[i].box.right_top.y, results.text_result[i].box.right_bottom.x, results.text_result[i].box.right_bottom.y, 255, 2);
            draw_line(&src_image, results.text_result[i].box.right_bottom.x, results.text_result[i].box.right_bottom.y, results.text_result[i].box.left_bottom.x, results.text_result[i].box.left_bottom.y, 255, 2);
            draw_line(&src_image, results.text_result[i].box.left_bottom.x, results.text_result[i].box.left_bottom.y, results.text_result[i].box.left_top.x, results.text_result[i].box.left_top.y, 255, 2);
            printf("regconize result: %s, score=%f\n", results.text_result[i].text.str, results.text_result[i].text.score);
        }
        cv::Mat result_mat = cv::Mat(src_image.height, src_image.width, CV_8UC3, src_image.virt_addr, src_image.width_stride);
        cv::imshow("out", result_mat);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            break;
        }
    }

    ret = release_ppocr_model(&rknn_app_ctx.det_context);
    if (ret != 0) {
        printf("release_ppocr_model det_context fail! ret=%d\n", ret);
    }

    ret = release_ppocr_model(&rknn_app_ctx.rec_context);
    if (ret != 0) {
        printf("release_ppocr_model rec_context fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL) {
        free(src_image.virt_addr);
    }

    return 0;
}
