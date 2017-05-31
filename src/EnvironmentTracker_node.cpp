#include <unistd.h>
#include <math.h>


#include <iostream>
#include <vector>

#include "ros/ros.h"
#include "mav_msgs/default_topics.h"
#include "mav_msgs/Actuators.h"
#include "mav_msgs/AttitudeThrust.h"
#include "mav_msgs/eigen_mav_msgs.h"
#include "geometry_msgs/Pose.h"
#include "std_srvs/Empty.h"
#include "gazebo_msgs/GetModelState.h"

#include "rotors_reinforce/PerformAction.h"

#include <sstream>

class environmentTracker {

private: ros::NodeHandle n;
    ros::Publisher firefly_motor_control_pub;
    ros::ServiceClient firefly_reset_client;
    ros::ServiceClient get_position_client;
    ros::Subscriber firefly_position_sub;
    ros::ServiceServer perform_action_srv;

public:
    double current_position[3];
    double current_orientation[4];
    double current_rotor_speed[6];

    environmentTracker(ros::NodeHandle node) {
        n = node;
        firefly_motor_control_pub = n.advertise<mav_msgs::Actuators>("/firefly/command/motor_speed", 1000);

        firefly_reset_client = n.serviceClient<std_srvs::Empty>("/gazebo/reset_world");
        get_position_client = n.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
     //   firefly_position_sub = n.subscribe("/firefly/ground_truth/pose", 1, &environmentTracker::poseCallback, this);
        perform_action_srv = n.advertiseService("env_tr_perform_action", &environmentTracker::performAction, this);

        current_rotor_speed = {0,0,0,0,0,0};
        ros::Rate loop_rate(100);
    }

    void respawn() {
        mav_msgs::Actuators msg;
        msg.angular_velocities = {0, 0, 0, 0, 0, 0};
        std_srvs::Empty srv;
        current_rotor_speed = {0,0,0,0,0,0};

        firefly_reset_client.call(srv);
        firefly_motor_control_pub.publish(msg);
    }

    bool performAction(rotors_reinforce::PerformAction::Request  &req, rotors_reinforce::PerformAction::Response &res) {

        for(int i = 0; i < sizeof(req.action); i++) {
            if (req.action[i] > 0) {
                current_rotor_speed = current_rotor_speed +10;
            }
            else {
                current_rotor_speed = current_rotor_speed -10;
            }
        }

        mav_msgs::Actuators msg;
        msg.angular_velocities = current_motor_speed;
        firefly_motor_control_pub.publish(msg);

        usleep(50);
        return true;
    }

//    void poseCallback(const geometry_msgs::Pose::ConstPtr& msg) {
//        current_position[0] = msg->position.x;
//        current_position[1] = msg->position.y;
//        current_position[2] = msg->position.z;
//        current_orientation[0] = msg->orientation.x;
//        current_orientation[1] = msg->orientation.y;
//        current_orientation[2] = msg->orientation.z;
//        current_orientation[3] = msg->orientation.w;

//        ROS_INFO("Position: [%f, %f, %f]", msg->position.x,
//                 msg->position.y,
//                 msg->position.z);
//        ROS_INFO("Orientation: [%f, %f, %f, %f]", msg->orientation.x,
//                 msg->orientation.y,
//                 msg->orientation.z,
//                 msg->orientation.w);
//    }

    void getPosition() {
            gazebo_msgs::GetModelState srv;
            srv.request.model_name = "firefly";
            if (get_position_client.call(srv)) {
                ROS_INFO("Position: %f %f %f", (float)srv.response.pose.position.x, (float)srv.response.pose.position.y, (float)srv.response.pose.position.z);
                current_position[0] = (float)srv.response.pose.position.x;
                current_position[1] = (float)srv.response.pose.position.y;
                current_position[2] = (float)srv.response.pose.position.z;
                current_orientation[0] = (float)srv.response.pose.orientation.x;
                current_orientation[1] = (float)srv.response.pose.orientation.y;
                current_orientation[2] = (float)srv.response.pose.orientation.z;
                current_orientation[3] = (float)srv.response.pose.orientation.w;
            }
            else {
                ROS_ERROR("Failed to get position");
            }
        }

    double getReward(const double targetx, const double targety, const double targetz, const int count) {


        double difx = current_position[0] - targetx;
        double dify = current_position[1] - targety;
        double difz = current_position[2] - targetz;

        if (current_position[2] <= 0.1 && count > 100) {
            return 0.0;
        }

        double reward4position =  1/(difx * difx + dify * dify + difz * difz + 1);
        double reward4orientation = 1/((current_orientation[0] * current_orientation[0] + current_orientation[1] * current_orientation[1] + current_orientation[2] * current_orientation[2])/(current_orientation[3] * current_orientation[3]) + 1);
        return reward4position * reward4orientation;
    }

};



int main(int argc, char **argv)
{


    ros::init(argc, argv, "talker");

    ros::NodeHandle n;

    ros::Rate loop_rate(100);

    environmentTracker* tracker = new environmentTracker(n);

    ros::spin();

     int count = 0;
     while (ros::ok)
     {
         if (tracker->current_position[2] <= 0.1 && count > 100) {
             count = 0;
             tracker->respawn();
         }

         ++count;
     }

    delete tracker;
    return 0;
}
