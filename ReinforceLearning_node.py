import rospy
from mav_msgs import Actuators

def talker():
    pub = rospy.Publisher('/firefly/command/motor_speed', Actuators, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        actuator_str = '{angular_velocities: [500, 500, 500, 500, 500, 500]}'
        rospy.loginfo(actuator_str)
        pub.publish(actuator_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
 
