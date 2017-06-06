#include <unistd.h>
#include <math.h>


#include <algorithm>
#include <iostream>
#include <vector>

#include "ros/ros.h"
#include "mav_msgs/default_topics.h"
#include "mav_msgs/RollPitchYawrateThrust.h"
#include "mav_msgs/Actuators.h"
#include "mav_msgs/eigen_mav_msgs.h"
#include "geometry_msgs/Pose.h"
#include "std_srvs/Empty.h"
#include "gazebo_msgs/GetModelState.h"
#include "gazebo_msgs/ContactsState.h"

#include "rotors_reinforce/PerformAction.h"
#include "rotors_reinforce/GetState.h"

#include <sstream>


int maxstep = 50;

const double max_v_xy = 1.0;  // [m/s]
const double max_roll = 10.0 * M_PI / 180.0;  // [rad]
const double max_pitch = 10.0 * M_PI / 180.0;  // [rad]
const double max_rate_yaw = 45.0 * M_PI / 180.0;  // [rad/s]
const double max_thrust = 30.0;  // [N]

const double axes_roll_direction = -1.0;
const double axes_pitch_direction = 1.0;
const double axes_thrust_direction = 1.0;

class environmentTracker {

private:
    ros::NodeHandle n;
    ros::Publisher firefly_control_pub;
    
    ros::ServiceClient firefly_reset_client;
    ros::ServiceClient get_position_client;
    ros::ServiceClient pause_physics;
    ros::ServiceClient unpause_physics;

    ros::Subscriber firefly_position_sub;
    ros::Subscriber firefly_collision_sub;
    ros::ServiceServer perform_action_srv;
    ros::ServiceServer get_state_srv;

    int step_counter;
    double current_yaw_vel_;
    bool crashed_flag;

public:
    std::vector<double> current_position;
    std::vector<double> current_orientation;
    std::vector<double> current_control_params;

    std::vector<double> target_position;

    environmentTracker(ros::NodeHandle node) {
    	current_position.resize(3);
        //hard constants for target
        target_position = {0.0, 0.0, 10.0};
    	current_orientation.resize(4);
        current_control_params.resize(4, 0);
    	step_counter = 0;

        current_yaw_vel_ = 0.0;

        n = node;
        firefly_control_pub = n.advertise<mav_msgs::RollPitchYawrateThrust>("/firefly/command/roll_pitch_yawrate_thrust", 1000);
        firefly_collision_sub = n.subscribe("/rotor_collision", 100, &environmentTracker::onCollision, this);
        firefly_reset_client = n.serviceClient<std_srvs::Empty>("/gazebo/reset_world");

        pause_physics = n.serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
        unpause_physics = n.serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");

        get_position_client = n.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
        perform_action_srv = n.advertiseService("env_tr_perform_action", &environmentTracker::performAction, this);
        get_state_srv = n.advertiseService("env_tr_get_state", &environmentTracker::getState, this);
    }

    void respawn() {
        mav_msgs::RollPitchYawrateThrust msg;
        msg.roll = 0;
        msg.pitch = 0;
        msg.yaw_rate = 0;
        msg.thrust.z = 0;
        std_srvs::Empty srv;

        current_position = {0,0,0};
        current_orientation = {0,0,0,0};
        step_counter = 0;

        current_control_params.resize(4, 0);

        firefly_reset_client.call(srv);
        firefly_control_pub.publish(msg);
    }

    void pausePhysics() {
    	std_srvs::Empty srv;
    	pause_physics.call(srv);
    }

    void unpausePhysics() {
    	std_srvs::Empty srv;
    	unpause_physics.call(srv);
    }

    void onCollision(const gazebo_msgs::ContactsState::ConstPtr& msg) {
        if (msg->states.size() > 1 || current_position[2] >= 50.0 || //z constraints
            current_position[1] >= 50.0 || current_position[1] <= -50.0 || //y constraints
            current_position[0] >= 50.0 || current_position[0] <= -50.0 //x constraints
            && step_counter > maxstep) {
                ROS_INFO("Crash, respawn...");
                step_counter = 0;
                crashed_flag = true;
                respawn();
        }
    }

    bool performAction(rotors_reinforce::PerformAction::Request  &req, rotors_reinforce::PerformAction::Response &res) {
        ROS_ASSERT(req.action.size() == current_control_params.size());

        step_counter++;

        // for(int i = 0; i < req.action.size()-1; i++) {
        //     if (req.action[i] == 1) { //1 =increase, 0 = stay, 2 = decrease
        //         current_control_params[i] = current_control_params[i] + 1.0;
        //         current_control_params[i] = std::min(current_control_params[i], 50.0);
        //     }
        //     if (req.action[i] == 2) {
        //         current_control_params[i] = current_control_params[i] - 1.0;
        //         current_control_params[i] = std::max(current_control_params[i], -50.0);
        //     }
        //     // switch(req.action[i]) {
            //     case 0:
            //         current_control_params[i] = -30.0;
            //         break;
            //     case 1:
            //         current_control_params[i] = -15.0;
            //         break;
            //     case 2:
            //         current_control_params[i] = 0.0;
            //         break;
            //     case 3:
            //         current_control_params[i] = 15.0;
            //         break;
            //     case 4: 
            //         current_control_params[i] = 30.0;
            //         break;
            //     }
        //}
       // current_control_params[3] = req.action[3] * 4;

        mav_msgs::RollPitchYawrateThrust msg;
        msg.roll = req.action[0] * max_roll * axes_roll_direction;
        msg.pitch = req.action[1] * max_pitch * axes_pitch_direction;

        if(req.action[2] > 0.01) {
            current_yaw_vel_ = max_rate_yaw;
        }
        else if (req.action[2] < -0.01) {
            current_yaw_vel_ = max_rate_yaw;   
        }
        else {
            current_yaw_vel_ = 0.0;   
        }

        msg.yaw_rate = current_yaw_vel_;
        msg.thrust.z = req.action[3] * max_thrust * axes_thrust_direction;

    
        ROS_INFO("roll: %f, pitch: %f, yaw_rate: %f, thrust %f", msg.roll, msg.pitch, msg.yaw_rate, msg.thrust.z);

        unpausePhysics();
        firefly_control_pub.publish(msg);
        usleep(50000);
    	getPosition();
        pausePhysics();

        res.target_position = target_position;
        res.position = current_position;
        res.orientation = current_orientation;
        res.reward = getReward(step_counter);

        //crash check at the end        
        if(crashed_flag) {
            res.crashed = true;
            crashed_flag = false;
        }

        return true;
    }

    bool getState(rotors_reinforce::GetState::Request &req, rotors_reinforce::GetState::Response &res) {
        getPosition();
        res.target_position = target_position;
        res.position = current_position;
        res.orientation = current_orientation;
        
        return true;
    }

    void getPosition() {
            gazebo_msgs::GetModelState srv;
            srv.request.model_name = "firefly";
            if (get_position_client.call(srv)) {
                ROS_INFO("Position: %f %f %f", (float)srv.response.pose.position.x, (float)srv.response.pose.position.y, (float)srv.response.pose.position.z);

                current_position[0] = (double)srv.response.pose.position.x;
                current_position[1] = (double)srv.response.pose.position.y;
                current_position[2] = (double)srv.response.pose.position.z;
                current_orientation[0] = (double)srv.response.pose.orientation.x;
                current_orientation[1] = (double)srv.response.pose.orientation.y;
                current_orientation[2] = (double)srv.response.pose.orientation.z;
                current_orientation[3] = (double)srv.response.pose.orientation.w;
            }
            else {
                ROS_ERROR("Failed to get position");
            }
    }

    double getReward(const int count) {
        double difx = current_position[0] - target_position[0];
        double dify = current_position[1] - target_position[1];
        double difz = current_position[2] - target_position[2];

        if (current_position[2] <= 0.1 && count > maxstep) {
            return 0.0;
        }

        //ROS_INFO("diffs^2: %f %f %f", difx * difx, dify * dify, difz * difz);
        double reward4position =  1/(difx * difx + dify * dify + difz * difz + 1.0);
        //double reward4orientation = 1/((current_orientation[0] * current_orientation[0] + current_orientation[1] * current_orientation[1] + current_orientation[2] * current_orientation[2])/(current_orientation[3] * current_orientation[3]) + 1);
        //return reward4position * reward4orientation;
        return reward4position;
    }

};



int main(int argc, char **argv)
{
    ros::init(argc, argv, "talker");

    ros::NodeHandle n;

    ros::Rate loop_rate(100);

    environmentTracker* tracker = new environmentTracker(n);

    if(argc == 2) {
        maxstep = atoi(argv[1]);
    }


    ROS_INFO("Comunication node ready");

    ros::spin();

    delete tracker;
    return 0;
}
