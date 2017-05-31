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

#define ROTORS_NUM 6

class environmentTracker {

private:
    ros::NodeHandle n;
    ros::Publisher firefly_motor_control_pub;
    
    ros::ServiceClient firefly_reset_client;
    ros::ServiceClient get_position_client;
    ros::ServiceClient pause_physics;
    ros::ServiceClient unpause_physics;

    ros::Subscriber firefly_position_sub;
    ros::ServiceServer perform_action_srv;

    int step_counter;

public:
    std::vector<double> current_position;
    std::vector<double> current_orientation;
    std::vector<double> current_rotor_speed;

    environmentTracker(ros::NodeHandle node) {
        current_position = {0,0,0};
        current_orientation = {0,0,0,0};
    	step_counter = 0;

        n = node;
        firefly_motor_control_pub = n.advertise<mav_msgs::Actuators>("/firefly/command/motor_speed", 1000);
        firefly_reset_client = n.serviceClient<std_srvs::Empty>("/gazebo/reset_world");

        pause_physics = n.serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
        unpause_physics = n.serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");

        get_position_client = n.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
        perform_action_srv = n.advertiseService("env_tr_perform_action", &environmentTracker::performAction, this);

        current_rotor_speed = {500,500,500,500,500,500};
        ros::Rate loop_rate(100);
    }

    void respawn() {
        mav_msgs::Actuators msg;
        msg.angular_velocities = {0, 0, 0, 0, 0, 0};
        std_srvs::Empty srv;
        current_rotor_speed = {500,500,500,500,500,500};
        current_position = {0,0,0};
        current_orientation = {0,0,0,0};
        step_counter = 0;

        firefly_reset_client.call(srv);
        firefly_motor_control_pub.publish(msg);
    }

    void pausePhysics() {
    	std_srvs::Empty srv;
    	pause_physics.call(srv);
    }

    void unpausePhysics() {
    	std_srvs::Empty srv;
    	unpause_physics.call(srv);
    }

    bool performAction(rotors_reinforce::PerformAction::Request  &req, rotors_reinforce::PerformAction::Response &res) {
    	ROS_ASSERT(req.action.size() == ROTORS_NUM);
        printf("received request...");

        step_counter++;

        if (current_position[2] <= 0.1 && step_counter > 100) {
                ROS_INFO("Crash, respawn...");
                respawn();
        }



        for(int i = 0; i < req.action.size(); i++) {
            if (req.action[i] > 0) {
                current_rotor_speed[i] = current_rotor_speed[i] +10;
            }
            else {
                current_rotor_speed[i] = current_rotor_speed[i] -10;
            }
        }

        mav_msgs::Actuators msg;
        msg.angular_velocities = current_rotor_speed;

        printf("publishing msg...");

        unpausePhysics();
    	firefly_motor_control_pub.publish(msg);
        usleep(50000);
    	getPosition();
        pausePhysics();

        printf("done.");

        printf("creating response...");
        res.position = current_position;
        res.orientation = current_orientation;
        res.reward = getReward(10, 10, 10, step_counter);
        printf("done.");
    }

    void getPosition() {
            printf("get position...");
            gazebo_msgs::GetModelState srv;
            srv.request.model_name = "firefly";
            if (get_position_client.call(srv)) {
                ROS_INFO("Position: %f %f %f", (float)srv.response.pose.position.x, (float)srv.response.pose.position.y, (float)srv.response.pose.position.z);
                printf("change curr pos, ori");
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
            printf("done.");
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

    delete tracker;
    return 0;
}
