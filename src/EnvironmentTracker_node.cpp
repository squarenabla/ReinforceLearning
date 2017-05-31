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

#include "reinforce_learning/PerformAction.h"

#include <sstream>

#define ROTORS_NUM 6

class environmentTracker {

private: ros::NodeHandle n;
    ros::Publisher firefly_motor_control_pub;
    ros::ServiceClient firefly_reset_client;
    ros::ServiceClient get_position_client;
    ros::Subscriber firefly_position_sub;
    ros::ServiceServer perform_action_srv;
    //ros::Publisher chatter_pub; 

public:
    std::vector<double> current_position;
    std::vector<double> current_orientation;

    environmentTracker(ros::NodeHandle node) {
    	current_position.resize(3);
    	current_orientation.resize(4);

        n = node;
        firefly_motor_control_pub = n.advertise<mav_msgs::Actuators>("/firefly/command/motor_speed", 1000);

        firefly_reset_client = n.serviceClient<std_srvs::Empty>("/gazebo/reset_world");
        get_position_client = n.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
     //   firefly_position_sub = n.subscribe("/firefly/ground_truth/pose", 1, &environmentTracker::poseCallback, this);
        perform_action_srv = n.advertiseService("env_tr_perform_action", &environmentTracker::performAction, this);

        ros::Rate loop_rate(100);
    }

    void respawn() {
        mav_msgs::Actuators msg;
        msg.angular_velocities = {0, 0, 0, 0, 0, 0};
        std_srvs::Empty srv;

        firefly_reset_client.call(srv);
        firefly_motor_control_pub.publish(msg);
    }

    bool performAction(reinforce_learning::PerformAction::Request  &req, reinforce_learning::PerformAction::Response &res) {
        //std::vector<int> actions(req.action);
    	ROS_ASSERT(req.action.size() == ROTORS_NUM);

    	//TODO: implement counter
    	int count = 30;

		if (current_position[2] <= 0.1 && count > 100) {
			ROS_INFO("Crash, respawn...");
			count = 0;
			respawn();
		}

		//TODO: map recieved actions to the real values
		//then send real values

        mav_msgs::Actuators msg;
    	msg.angular_velocities = {(float)req.action[0], (float)req.action[1], (float)req.action[2], (float)req.action[3], (float)req.action[4], (float)req.action[5]};

    	firefly_motor_control_pub.publish(msg);

    	usleep(100000);


    	getPosition();

    	res.position = current_position;
    	res.orientation = current_orientation;
    	res.reward = getReward(10, 10, 10, count);

        return true;
    }

    void poseCallback(const geometry_msgs::Pose::ConstPtr& msg) {
        current_position[0] = msg->position.x;
        current_position[1] = msg->position.y;
        current_position[2] = msg->position.z;
        current_orientation[0] = msg->orientation.x;
        current_orientation[1] = msg->orientation.y;
        current_orientation[2] = msg->orientation.z;
        current_orientation[3] = msg->orientation.w;

        ROS_INFO("Position: [%f, %f, %f]", msg->position.x,
                 msg->position.y,
                 msg->position.z);
        ROS_INFO("Orientation: [%f, %f, %f, %f]", msg->orientation.x,
                 msg->orientation.y,
                 msg->orientation.z,
                 msg->orientation.w);
    }

    void getPosition() {
        gazebo_msgs::GetModelState srv;
        srv.request.model_name = "firefly";
        if (get_position_client.call(srv)) {
            ROS_INFO("Possition: %f %f %f", (float)srv.response.pose.position.x, (float)srv.response.pose.position.y, (float)srv.response.pose.position.z);
        }
        else {
            ROS_ERROR("Failed to get position");
            return;
        }

        current_position[0] = (double)srv.response.pose.position.x;
        current_position[1] = (double)srv.response.pose.position.y;
        current_position[2] = (double)srv.response.pose.position.z;
        current_orientation[0] = (double)srv.response.pose.orientation.x;
        current_orientation[1] = (double)srv.response.pose.orientation.y;
        current_orientation[2] = (double)srv.response.pose.orientation.z;
        current_orientation[3] = (double)srv.response.pose.orientation.w;
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

    // int count = 0;
    // while (1)
    // {
    //     if (tracker->current_position[2] <= 0.1 && count > 100) {
    //         count = 0;
    //         tracker->respawn();
    //     }

    //     mav_msgs::Actuators msg;
    //     float velocity = atof(argv[1]);

    //     msg.angular_velocities = {velocity, velocity, velocity, velocity, velocity, velocity};

    //     chatter_pub.publish(msg);

    //     tracker->getPosition();
    //     usleep(100000);
    //     ++count;
    // }

    delete tracker;
    return 0;
}
