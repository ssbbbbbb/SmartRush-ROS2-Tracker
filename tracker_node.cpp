#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/float64.hpp"
#include "mavros_msgs/msg/state.hpp"
#include "mavros_msgs/msg/position_target.hpp"
#include "mavros_msgs/srv/command_bool.hpp"
#include "mavros_msgs/srv/set_mode.hpp"
#include "mavros_msgs/srv/command_tol.hpp"

using namespace std::chrono_literals;

struct PIDController {
    float kp, ki, kd;
    float integral = 0.0f;
    float last_error = 0.0f;
    float max_integral; 

    PIDController(float p, float i, float d, float max_i = 1.0f) 
        : kp(p), ki(i), kd(d), max_integral(max_i) {}

    float calculate(float error, float dt) {
        if (dt <= 0.0f) return 0.0f;
        integral += error * dt;
        integral = std::clamp(integral, -max_integral, max_integral);
        float derivative = (error - last_error) / dt;
        last_error = error;
        return (kp * error) + (ki * integral) + (kd * derivative);
    }

    void reset() {
        integral = 0.0f;
        last_error = 0.0f;
    }
};

class DroneTracker : public rclcpp::Node {
public:
    DroneTracker() : Node("drone_tracker_node"),
                     pid_vx_size_(4.0f, 0.2f, 3.0f, 1.0f), 
                     pid_vx_pos_(2.5f, 0.1f, 1.0f, 0.8f),
                     pid_yaw_(2.2f, 0.15f, 1.2f, 0.6f) {
                         
        std::string pkg_path = ament_index_cpp::get_package_share_directory("drone_tracker");
        std::string model_path = pkg_path + "/models/yolov5n.onnx";
        net_ = cv::dnn::readNetFromONNX(model_path);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        state_sub_ = this->create_subscription<mavros_msgs::msg::State>(
            "/mavros/state", 10, [this](const mavros_msgs::msg::State::SharedPtr msg) { current_state_ = *msg; });
        
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/iris/camera/image_raw", 10, std::bind(&DroneTracker::image_callback, this, std::placeholders::_1));

        vel_pub_ = this->create_publisher<mavros_msgs::msg::PositionTarget>("/mavros/setpoint_raw/local", 10);
        gimbal_pub_ = this->create_publisher<std_msgs::msg::Float64>("/gimbal/cmd_pitch", 10);

        arm_client_ = this->create_client<mavros_msgs::srv::CommandBool>("/mavros/cmd/arming");
        mode_client_ = this->create_client<mavros_msgs::srv::SetMode>("/mavros/set_mode");
        takeoff_client_ = this->create_client<mavros_msgs::srv::CommandTOL>("/mavros/cmd/takeoff");

        timer_ = this->create_wall_timer(100ms, std::bind(&DroneTracker::control_loop, this));
        last_time_ = this->now();

        RCLCPP_INFO(this->get_logger(), "🛡️ [預測補償增強模式] 啟動！目標消失預測已開啟。");
    }

private:
    // --- 調整後的參數 ---
    const float TARGET_HEIGHT_RATIO = 0.28f; 
    const float VX_MAX = 5.0f;               
    const float YAW_MAX = 2.5f;              
    const float DEADZONE_H = 0.01f;          
    const float SEARCH_YAW = 0.7f;

    // --- 預測補償參數 ---
    const float PANIC_CLOSE_THRESHOLD = 0.85f; // 畫面下方 85% 為危險區
    const float PANIC_VX_BOOST = -3.5f;        // 進入危險區強制後退速度
    const float LOST_RUSH_TIME = 1.5f;         // 目標消失後的盲衝時間 (秒)
    const float NORMAL_ACCEL_LIMIT = 0.40f;  

    float last_vx_cmd_ = 0.0f;               
    float last_size_error_ = 0.0f;
    
    // 消失前的最後位置快照
    float last_known_cx_ = 0.5f;
    float last_known_cy_ = 0.5f;
    rclcpp::Time lost_time_;

    PIDController pid_vx_size_;
    PIDController pid_vx_pos_;
    PIDController pid_yaw_;

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        if (state_ == "WAITING" || state_ == "TAKEOFF" || state_ == "CLIMBING") return;

        try {
            cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;
            int img_w = frame.cols, img_h = frame.rows;
            cv::Rect p_box;

            rclcpp::Time current_time = this->now();
            float dt = (current_time - last_time_).seconds();
            last_time_ = current_time;

            if (find_best_person(frame, p_box)) {
                state_ = "TRACKING";
                lost_target_counter_ = 0;

                float p_cx_norm = (p_box.x + p_box.width / 2.0f) / img_w;
                float p_cy_norm = (p_box.y + p_box.height / 2.0f) / img_h;
                float p_h_ratio = (float)p_box.height / img_h;

                // 記錄最後位置供消失預測用
                last_known_cx_ = p_cx_norm;
                last_known_cy_ = p_cy_norm;

                auto target_msg = create_body_vel_msg();

                // --- 1. 計算 Vx 原始需求 ---
                float size_error = TARGET_HEIGHT_RATIO - p_h_ratio;
                float vx_by_size = pid_vx_size_.calculate(size_error, dt);
                float pos_v_error = 0.5f - p_cy_norm;
                float vx_by_pos = pid_vx_pos_.calculate(pos_v_error, dt);

                float raw_vx = vx_by_size + vx_by_pos;

                // --- 2. 補償邏輯：太近(太下面)則加速補償 ---
                if (p_cy_norm > PANIC_CLOSE_THRESHOLD) {
                    raw_vx = PANIC_VX_BOOST; // 強制全速後退
                } else if (raw_vx < 0) {
                    raw_vx *= 2.0f; // 常規後退強化
                }

                // 加速度限制
                float vx_diff = raw_vx - last_vx_cmd_;
                float current_limit = (p_cy_norm > PANIC_CLOSE_THRESHOLD) ? 2.0f : NORMAL_ACCEL_LIMIT;
                vx_diff = std::clamp(vx_diff, -current_limit, current_limit);
                float final_vx = last_vx_cmd_ + vx_diff;
                
                last_vx_cmd_ = final_vx;
                target_msg.velocity.x = std::clamp(final_vx, -VX_MAX, VX_MAX);

                // --- 3. 轉向控制 ---
                float pos_h_error = 0.5f - p_cx_norm;
                float yaw_rate = pid_yaw_.calculate(pos_h_error, dt);
                target_msg.yaw_rate = std::clamp(yaw_rate, -YAW_MAX, YAW_MAX);

                vel_pub_->publish(target_msg);
                send_gimbal_cmd(0.5236f); 
                draw_debug_info(frame, p_box, target_msg.velocity.x, target_msg.yaw_rate, p_h_ratio);

            } else {
                handle_lost_target_with_prediction();
            }
            cv::imshow("Pro Stable Tracker", frame);
            cv::waitKey(1);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV Error: %s", e.what());
        }
    }

    // --- 核心邏輯：目標消失後的預測衝刺 ---
    void handle_lost_target_with_prediction() {
        if (state_ == "TRACKING") {
            state_ = "LOST_RUSH";
            lost_time_ = this->now();
            RCLCPP_WARN(this->get_logger(), "Target Lost! Entering PREDICTION RUSH.");
        }

        if (state_ == "LOST_RUSH") {
            float elapsed = (this->now() - lost_time_).seconds();
            if (elapsed < LOST_RUSH_TIME) {
                auto rush_msg = create_body_vel_msg();

                // 根據消失前的最後位置判定衝刺方向
                // 1. 如果在左/右邊邊緣消失，保持原本旋轉方向
                if (last_known_cx_ < 0.2f) rush_msg.yaw_rate = YAW_MAX * 0.8f;
                else if (last_known_cx_ > 0.8f) rush_msg.yaw_rate = -YAW_MAX * 0.8f;

                // 2. 如果在底部消失，給予持續後退衝量
                if (last_known_cy_ > 0.7f) rush_msg.velocity.x = -2.5f;
                // 3. 如果在頂部消失(跑遠了)，給予向前衝量
                else if (last_known_cy_ < 0.3f) rush_msg.velocity.x = 2.0f;

                vel_pub_->publish(rush_msg);
            } else {
                state_ = "SEARCHING";
                last_vx_cmd_ = 0.0f;
            }
        }
    }

    // (find_best_person, draw_debug_info, create_body_vel_msg, control_loop 等保持原本邏輯...)
    // 這裡為了簡潔省略，請沿用你上一版的實作，但確保 state_ 判定包含 "LOST_RUSH"
    
    // ... 原本的 find_best_person ...
    bool find_best_person(const cv::Mat& frame, cv::Rect& best_box) {
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        net_.setInput(blob);
        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());
        float* data = (float*)outputs[0].data;
        int img_w = frame.cols, img_h = frame.rows;
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        for (int i = 0; i < 25200; ++i) {
            float* row = data + i * 85;
            if (row[4] > 0.5f) {
                if (row[5] > 0.45f) { // 假設 Index 0 是人
                    int w = row[2] * img_w / 640; int h = row[3] * img_h / 640;
                    boxes.push_back(cv::Rect(row[0]*img_w/640 - w/2, row[1]*img_h/640 - h/2, w, h));
                    confidences.push_back(row[4] * row[5]);
                }
            }
        }
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, 0.5f, 0.4f, indices);
        if (indices.empty()) return false;
        best_box = boxes[indices[0]];
        return true;
    }

    void draw_debug_info(cv::Mat& frame, cv::Rect box, float vx, float yaw, float h_ratio) {
        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        std::string info = "STATE: " + state_ + " Vx: " + std::to_string(vx).substr(0,4);
        cv::putText(frame, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    }

    mavros_msgs::msg::PositionTarget create_body_vel_msg() {
        mavros_msgs::msg::PositionTarget msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = "base_link"; 
        msg.coordinate_frame = mavros_msgs::msg::PositionTarget::FRAME_BODY_NED;
        msg.type_mask = 0b011111000111; 
        msg.velocity.x = 0.0; msg.velocity.y = 0.0; msg.velocity.z = 0.0;
        msg.yaw_rate = 0.0;
        return msg;
    }

    void control_loop() {
        if (!current_state_.connected) return;
        if (state_ == "WAITING") {
            if (current_state_.mode != "GUIDED") set_mode("GUIDED");
            else if (!current_state_.armed) arm_drone(true);
            else state_ = "TAKEOFF";
        } 
        else if (state_ == "TAKEOFF") {
            send_takeoff(5.0); state_ = "CLIMBING"; start_time_ = this->now();
        } 
        else if (state_ == "CLIMBING") {
            send_gimbal_cmd(0.5236f); 
            if ((this->now() - start_time_).seconds() > 8.0) state_ = "SEARCHING";
        } 
        else if (state_ == "SEARCHING") {
            auto rot_msg = create_body_vel_msg();
            rot_msg.yaw_rate = SEARCH_YAW; vel_pub_->publish(rot_msg);
        }
    }

    void send_gimbal_cmd(float pitch) {
        std_msgs::msg::Float64 msg; msg.data = (double)pitch; gimbal_pub_->publish(msg);
    }
    void set_mode(std::string mode) {
        auto req = std::make_shared<mavros_msgs::srv::SetMode::Request>();
        req->custom_mode = mode; mode_client_->async_send_request(req);
    }
    void arm_drone(bool arm) {
        auto req = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
        req->value = arm; arm_client_->async_send_request(req);
    }
    void send_takeoff(float alt) {
        auto req = std::make_shared<mavros_msgs::srv::CommandTOL::Request>();
        req->altitude = alt; takeoff_client_->async_send_request(req);
    }

    std::string state_ = "WAITING";
    mavros_msgs::msg::State current_state_;
    rclcpp::Time start_time_, last_time_;
    cv::dnn::Net net_;
    int lost_target_counter_ = 0;

    rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<mavros_msgs::msg::PositionTarget>::SharedPtr vel_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr gimbal_pub_;
    rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr arm_client_;
    rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr mode_client_;
    rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedPtr takeoff_client_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DroneTracker>());
    rclcpp::shutdown();
    return 0;
}