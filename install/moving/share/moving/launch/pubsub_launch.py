
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    computation_controller_node = Node(
        package='pubsub_turtle',        
        executable='computation_controller',  
        name='computation_controller',
        output='screen'
    )


    move_controller_node = Node(
        package='pubsub_turtle',        
        executable='move_controller',   
        name='move_controller',
        output='screen'
    )

    return LaunchDescription([
        computation_controller_node,
        move_controller_node
    ])
