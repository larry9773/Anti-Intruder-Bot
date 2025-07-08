import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import LaserScan, Image, Imu
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
import heapq
from irobot_create_msgs.action import Undock

class BallDetector:
    def __init__(self):
        self.lower1 = np.array([0,120,70]); self.upper1 = np.array([10,255,255])
        self.lower2 = np.array([170,120,70]); self.upper2 = np.array([180,255,255])
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    def detect(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, self.lower1, self.upper1)
        m2 = cv2.inRange(hsv, self.lower2, self.upper2)
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print('[BallDetector] No contours found')
            return None, None
        c = max(contours, key=cv2.contourArea)
        (x,y), r = cv2.minEnclosingCircle(c)
        if r < 5:
            print(f'[BallDetector] Detected contour radius too small: {r}')
            return None, None
        print(f'[BallDetector] Ball detected at ({int(x)},{int(y)}) with radius {r}')
        return (int(x),int(y)), r

class OccupancyGrid:
    def __init__(self, size=20.0, res=0.1):
        self.res = res
        self.dim = int(size/res)
        self.grid = -np.ones((self.dim,self.dim), np.int8)
        self.origin = np.array([self.dim//2,self.dim//2])
    def world_to_idx(self, x,y):
        j=int(round(x/self.res))+self.origin[0]
        i=int(round(y/self.res))+self.origin[1]
        return i,j
    def update(self, pose, scan):
        x0,y0,theta = pose
        ranges = np.array(scan.ranges)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        for r,ang in zip(ranges,angles):
            if np.isfinite(r) and r<scan.range_max:
                x = x0 + r * math.cos(theta + ang)
                y = y0 + r * math.sin(theta + ang)
                i,j = self.world_to_idx(x,y)
                if 0<=i<self.dim and 0<=j<self.dim:
                    self.grid[i,j] = 1
        i0,j0 = self.world_to_idx(x0,y0)
        self.grid[i0,j0] = 0
        print('[OccupancyGrid] Map updated at pose', pose)
    def find_frontiers(self):
        fr = []
        for i in range(1,self.dim-1):
            for j in range(1,self.dim-1):
                if self.grid[i,j] == 0:
                    neigh = self.grid[i-1:i+2, j-1:j+2]
                    if np.any(neigh == -1): fr.append((i,j))
        print(f'[OccupancyGrid] Found {len(fr)} frontier cells')
        return fr

def astar(start, goal, grid):
    print(f'[A*] Planning from {start} to {goal}')
    def h(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
    open = [(h(start,goal),0,start,None)]; came = {}; cost={start:0}
    while open:
        _,c,u,p = heapq.heappop(open)
        if u in came: continue
        came[u] = p
        if u == goal: break
        for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            v=(u[0]+di, u[1]+dj)
            if 0<=v[0]<grid.shape[0] and 0<=v[1]<grid.shape[1] and grid[v]==0:
                nc = c+1
                if v not in cost or nc<cost[v]:
                    cost[v] = nc
                    heapq.heappush(open, (nc+h(v,goal), nc, v, u))
    if goal not in came:
        print('[A*] No path found')
        return None
    path=[]; u=goal
    while u:
        path.append(u)
        u=came[u]
    path = path[::-1]
    print(f'[A*] Path found with length {len(path)}')
    return path

class PurePursuit:
    def __init__(self, lookahead=5): self.ld=lookahead
    def run(self, pose, path, grid):
        pts=[((j-grid.origin[0])*grid.res, (i-grid.origin[1])*grid.res) for i,j in path]
        ds=[math.hypot(px-pose[0], py-pose[1]) for px,py in pts]
        idx = min(range(len(ds)), key=lambda k: abs(ds[k]-self.ld))
        tx,ty = pts[idx]
        alpha = math.atan2(ty-pose[1], tx-pose[0]) - pose[2]
        lin = min(0.3, ds[idx])
        ang = max(-1, min(1, 2*alpha))
        print(f'[PurePursuit] Following path point {idx}, lin={lin}, ang={ang}')
        return lin, ang

class AutonomousNavigator(Node):
    def __init__(self):
        super().__init__('autonomous_navigator')
        print('[Navigator] Initializing')
        self.bridge = CvBridge(); self.detector = BallDetector()
        self.grid = OccupancyGrid(size=20.0, res=0.1)
        self.pp = PurePursuit(lookahead=10)
        hfov = math.radians(69.0)
        self.img_w = 640
        self.img_h = 480
        self.fx = self.img_w/(2*math.tan(hfov/2))
        self.last_image = None; self.last_scan = None
        self.pose=[0.0,0.0,0.0]
        self.last_time=None
        self.path = None; self.undocked = False
        self.undock_client = ActionClient(self, Undock, 'undock')
        self.create_subscription(Image, 'oakd/rgb/preview/image_raw', self.img_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Imu, '/imu', self.imu_cb, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.1, self.timer_cb)
        self.ball_R = 0.20  # Ball radius in meters


    def perform_undock(self):
        print('[Navigator] Performing undock')
        if not self.undock_client.wait_for_server(timeout_sec=5.0):
            print('Error: Undock server not available')
            rclpy.shutdown()
            return False
        goal_future = self.undock_client.send_goal_async(Undock.Goal())
        rclpy.spin_until_future_complete(self, goal_future, timeout_sec=5.0)
        if not goal_future.done():
            print('Error: Undock goal timeout')
            rclpy.shutdown()
            return False
        handle = goal_future.result()
        if not handle.accepted:
            print('Error: Undock goal rejected')
            rclpy.shutdown()
            return False
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=10.0)
        if not result_future.done() or not result_future.result().result.success:
            print('Error: Undock failed')
            rclpy.shutdown()
            return False
        print('[Navigator] Undock completed')
        self.undocked = True
        return True
    def img_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.last_image = cv2.resize(img, (self.img_w, self.img_h))
        print('[Navigator] Image received')
    def scan_cb(self, msg):
        self.last_scan = msg
        print('[Navigator] Scan received')
    def imu_cb(self, msg):
        q=msg.orientation; siny=2.0*(q.w*q.z+q.x*q.y)
        cosy=1.0-2.0*(q.y*q.y+q.z*q.z); self.pose[2]=math.atan2(siny,cosy)
        print(f'[Navigator] IMU yaw updated: {self.pose[2]}')
    def timer_cb(self):
        print(f'[Navigator] Timer tick, undocked={self.undocked}')
        if not self.undocked:
            if not self.perform_undock(): return
        now=self.get_clock().now()
        if self.last_time is None: self.last_time=now
        if self.last_image is None or self.last_scan is None:
            print('[Navigator] Waiting for sensor data')
            self.last_time=now
            return
        dt=(now-self.last_time).nanoseconds*1e-9
        if self.path:
            lin,ang=self.pp.run(self.pose,self.path,self.grid)
        else:
            center,r=self.detector.detect(self.last_image)
            if center:
                print('[Navigator] Ball detected, planning path')
                theta=math.atan2((center[0]-self.img_w/2),self.fx)
                dist=(self.fx*self.ball_R)/r
                xg=self.pose[0]+dist*math.cos(self.pose[2]+theta)
                yg=self.pose[1]+dist*math.sin(self.pose[2]+theta)
                goal=self.grid.world_to_idx(xg,yg)
                start=self.grid.world_to_idx(self.pose[0],self.pose[1])
                p=astar(start,goal,self.grid.grid)
                if p:
                    self.path=p
        if not self.path:
            lin,ang=0.15,0.5
        cmd=Twist(); cmd.linear.x=lin; cmd.angular.z=ang; self.pub.publish(cmd)
        print(f'[Navigator] Command published: lin={lin}, ang={ang}')
        self.grid.update(self.pose,self.last_scan)
        self.pose[0]+=lin*dt*math.cos(self.pose[2]); self.pose[1]+=lin*dt*math.sin(self.pose[2])
        self.last_time=now

def main(args=None):
    rclpy.init(args=args)
    node=AutonomousNavigator()
    print('[Main] Spinning node')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    print('[Main] Shutdown complete')
