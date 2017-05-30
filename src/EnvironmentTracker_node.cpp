#include "ros/ros.h"
#include "mav_msgs/default_topics.h"
#include "mav_msgs/Actuators.h"
#include "mav_msgs/AttitudeThrust.h"
#include "mav_msgs/eigen_mav_msgs.h"
#include "geometry_msgs/Pose.h"
#include "std_srvs/Empty.h"

#include <sstream>

class environmentTracker {

private: ros::NodeHandle n;
    ros::Publisher firefly_motor_control_pub;
    ros::Publisher firefly_control_pub;
    ros::ServiceClient firefly_reset_client;
    ros::Subscriber firefly_position_sub;
public:
    float current_position[3];
    float current_orientation[4];

    environmentTracker(ros::NodeHandle node) {
        n = node;
        firefly_motor_control_pub = n.advertise<mav_msgs::Actuators>("/firefly/command/motor_speed", 1000);

        firefly_control_pub = n.advertise<mav_msgs::RollPitchYawrateThrust>(mav_msgs::default_topics::COMMAND_ROLL_PITCH_YAWRATE_THRUST, 10);

        firefly_reset_client = n.serviceClient<std_srvs::Empty>("/gazebo/reset_world");

        firefly_position_sub = n.subscribe("/firefly/ground_truth/pose", 1, &environmentTracker::poseCallback, this);

        ros::Rate loop_rate(100);
    }

    void executeAction(Action a) {

    }

    void respawn() {
        mav_msgs::Actuators msg;
        msg.angular_velocities = {0, 0, 0, 0, 0, 0};
        std_srvs::Empty srv;

        firefly_reset_client.call(srv);
        firefly_motor_control_pub.publish(msg);
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

};



int main(int argc, char **argv)
{
    ros::init(argc, argv, "environmentTracker");

    ros::NodeHandle n;

    environmentTracker* tracker = new environmentTracker(n);

    ros::Rate loop_rate(100);

    mav_msgs::RollPitchYawrateThrust control_msg;
    float velocity = atof(argv[1]);
    float thrustx = atof(argv[2]);
    control_msg.roll = 1;
    control_msg.pitch = velocity;
    control_msg.yaw_rate = 1;
    control_msg.thrust.x = thrustx;
    control_msg.thrust.y = 0;
    control_msg.thrust.z = 0;

    int count = 0;
    while (ros::ok())
    {
        if (tracker->current_position[2] <= 0.01 && count > 100) {
            count = 0;
            tracker->respawn();
        }    

        tracker->executeAction(control_msg);

        ros::spinOnce();

        loop_rate.sleep();
        ++count;
    }


    return 0;
}
