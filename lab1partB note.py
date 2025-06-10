import sys
import heapq
import pandas as pd
import matplotlib.pyplot as plt

# ========= 常量定义 =========
LINK_RATE_BPS = 10_000_000_000  # 固定链路速率为10 Gbps（即10^10 bps）

# ========= 事件类定义 =========
class Event:
    def __init__(self, time, type, pkt_id):
        self.time = time      # 事件发生的时间（单位：微秒 us）
        self.type = type      # 事件类型：'arrival' 或 'departure'
        self.pkt_id = pkt_id  # 包的唯一编号

    def __lt__(self, other):
        # 让事件对象支持按时间排序（用于 heapq 优先队列）
        return self.time < other.time

# ========= 主模拟器类 =========
class TraceSimulator:
    def __init__(self, trace_file):
        # 读取输入的 trace 文件（每一行为： inter-arrival(us) pkt_size(bytes)）
        self.trace = pd.read_csv(trace_file, sep=r'\s+', header=None, names=['inter_arrival_us', 'pkt_size_bytes'])
        print(f"Loaded trace with {len(self.trace)} packets.")

        # 初始化状态
        self.queue = []                  # 排队队列，存储 (pkt_id, 到达时间, 服务时间)
        self.event_list = []            # 优先队列，事件按时间排序
        self.curr_time = 0.0            # 当前仿真时间（us）
        self.in_service = False         # 是否有包正在服务
        self.total_delay = 0.0          # 总延迟时间累加（用于计算平均时延）
        self.total_pkts = 0             # 总处理包数
        self.queue_len_record = [0] * 12  # 到达时系统中包数量的分布 P(n)，最多记录到 >10
        self.pkts_in_system = 0         # 当前系统中包的数量（= 队列 + 正在服务的1个）
        self.area_under_q = 0.0         # 积分区域：用于计算平均系统内包数 N
        self.last_event_time = 0.0      # 上一个事件时间（用于计算积分）

    # 添加事件到事件列表中（heapq实现的最小堆）
    def schedule_event(self, event):
        heapq.heappush(self.event_list, event)

    # ========= 主模拟过程 =========
    def run(self):
        # 首先将所有到达事件调度入事件列表
        arrival_time = 0.0
        for pkt_id, row in self.trace.iterrows():
            inter_arrival = row['inter_arrival_us']
            arrival_time += inter_arrival
            self.schedule_event(Event(arrival_time, 'arrival', pkt_id))

        if not self.event_list:
            print("No events scheduled.")
            return

        print(f"First event scheduled at {self.event_list[0].time:.2f} us")
        print("Starting simulation loop...")

        # 循环处理事件队列，直到处理完所有事件
        while self.event_list:
            event = heapq.heappop(self.event_list)
            self.curr_time = event.time

            # 使用梯形积分方式更新 Q(t) 曲线面积
            self.area_under_q += self.pkts_in_system * (self.curr_time - self.last_event_time)
            self.last_event_time = self.curr_time

            # 调用对应事件处理函数
            if event.type == 'arrival':
                self.handle_arrival(event)
            elif event.type == 'departure':
                self.handle_departure(event)

    # ========= 到达事件处理函数 =========
    def handle_arrival(self, event):
        # 读取该包大小（单位：字节），转化为服务时间（us）
        pkt_size_bytes = self.trace.loc[event.pkt_id, 'pkt_size_bytes']
        service_time_us = (pkt_size_bytes * 8 / LINK_RATE_BPS) * 1e6  # us

        print(f"[{self.curr_time:.2f}]: pkt {event.pkt_id} arrives and finds {len(self.queue)} packets in the queue")

        # 到达时记录系统中包数（用于估计 P(n)）
        index = self.pkts_in_system if self.pkts_in_system <= 10 else 11
        self.queue_len_record[index] += 1
        self.pkts_in_system += 1

        # 把该包加入队列：记录 pkt_id，到达时间，服务时间
        self.queue.append((event.pkt_id, self.curr_time, service_time_us))

        # 如果当前没有包在服务中，就立即开始服务
        if not self.in_service:
            self.start_service()

    # ========= 离开事件处理函数 =========
    def handle_departure(self, event):
        # 从队列头部移除正在服务的包
        pkt_id, arrival_time, _ = self.queue.pop(0)
        delay = self.curr_time - arrival_time

        print(f"[{self.curr_time:.2f}]: pkt {pkt_id} departs having spent {delay:.2f} us in the system")

        self.total_delay += delay
        self.total_pkts += 1
        self.pkts_in_system -= 1

        # 如果还有排队的包，开始服务下一个；否则空闲
        if self.queue:
            self.start_service()
        else:
            self.in_service = False

    # ========= 启动下一个包的服务 =========
    def start_service(self):
        self.in_service = True
        pkt_id, _, service_time = self.queue[0]
        departure_time = self.curr_time + service_time
        self.schedule_event(Event(departure_time, 'departure', pkt_id))

    # ========= 打印统计结果 =========
    def print_stats(self):
        # 平均系统内包数 N = 积分面积 / 总时间
        N = self.area_under_q / self.curr_time

        # 平均系统时延 T = 总延迟 / 包数量
        T = self.total_delay / self.total_pkts

        print(f"\nSimulation Results (fixed 10 Gbps system):")
        print(f"Total packets simulated: {self.total_pkts}")
        print(f"Average number in system (N): {N:.4f}")
        print(f"Average time in system (T): {T:.4f} us")

        # 打印 P(n) 概率分布
        total_arrivals = sum(self.queue_len_record)
        print("\nP(n) distribution (based on arrivals):")
        pn_probs = []
        for n in range(11):
            p = self.queue_len_record[n] / total_arrivals
            pn_probs.append(p)
            print(f"P({n}) = {p:.4f}")
        p_over_10 = self.queue_len_record[11] / total_arrivals
        pn_probs.append(p_over_10)
        print(f"P(n > 10) = {p_over_10:.4f}")
        print(f"[Check] Total probability = {sum(pn_probs):.6f}")

        # 绘制 P(n) 柱状图
        labels = [str(i) for i in range(11)] + ['>10']
        plt.figure(figsize=(10, 6))
        plt.bar(labels, pn_probs)
        plt.xlabel("Number of packets in system (n)")
        plt.ylabel("Probability P(n)")
        plt.title("P(n) Distribution Based on Packet Arrivals")
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

# ========= 程序主入口 =========
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Lab01.py <trace_file>")
        sys.exit(1)

    trace_file = sys.argv[1]  # 从命令行读取 trace 文件路径
    sim = TraceSimulator(trace_file)
    sim.run()
    sim.print_stats()
