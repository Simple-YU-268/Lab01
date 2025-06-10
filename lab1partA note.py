import sys
import heapq
import random
import matplotlib.pyplot as plt

# ========== 全局配置参数 ==========
MEAN_PKT_SIZE_BITS = 10000               # 平均包大小（比特）
LINK_RATE_BPS = 10_000_000_000           # 链路速率（比特/秒） = 10 Gbps

# ========== 定义事件类 ==========
class Event:
    def __init__(self, time, type, pkt_id):
        self.time = time          # 事件时间
        self.type = type          # 'arrival' 到达 或 'departure' 离开
        self.pkt_id = pkt_id      # 包的唯一标识符（ID）

    def __lt__(self, other):
        # 定义优先队列排序方式，按时间升序排列
        return self.time < other.time

# ========== 主模拟器类 ==========
class MM1Simulator:
    def __init__(self, npkts, lambd):
        self.npkts = npkts                  # 模拟的总包数（终止条件）
        self.lambd = lambd                  # 包到达率 λ，单位是每微秒
        self.queue = []                     # 排队的包 [(pkt_id, 到达时间)]
        self.event_list = []                # 事件优先队列（heap）
        self.curr_time = 0.0                # 当前仿真时间
        self.next_pkt_id = 0                # 下一个包的编号
        self.in_service = False             # 是否有包正在被服务
        self.total_delay = 0.0              # 所有包累计延迟（us）
        self.total_pkts = 0                 # 已处理完成的包数
        self.queue_len_record = [0] * 12    # 用于记录每个n值出现次数（0-10 + >10）
        self.pkts_in_system = 0             # 当前系统中包数（队列+服务中）
        self.area_under_q = 0.0             # 用于计算平均系统长度 N
        self.last_event_time = 0.0          # 上一个事件发生时间（用于积分）

    def schedule_event(self, event):
        # 将事件加入优先队列
        heapq.heappush(self.event_list, event)

    def run(self):
        # 生成第一个到达事件
        first_arrival_time = random.expovariate(self.lambd)
        self.schedule_event(Event(first_arrival_time, 'arrival', self.next_pkt_id))
        self.next_pkt_id += 1

        # 主循环，直到完成 npkts 个包
        while self.total_pkts < self.npkts:
            event = heapq.heappop(self.event_list)
            self.curr_time = event.time

            # 积分计算：面积 = 当前系统内包数 × 时间间隔
            self.area_under_q += self.pkts_in_system * (self.curr_time - self.last_event_time)
            self.last_event_time = self.curr_time

            # 处理事件
            if event.type == 'arrival':
                self.handle_arrival(event)
            elif event.type == 'departure':
                self.handle_departure(event)

    def handle_arrival(self, event):
        # 到达时打印信息
        print(f"[{event.time:.2f}]: pkt {event.pkt_id} arrives and finds {len(self.queue)} packets in the queue")

        # 记录系统中当前包数用于估计 P(n)
        index = self.pkts_in_system if self.pkts_in_system <= 10 else 11
        self.queue_len_record[index] += 1
        self.pkts_in_system += 1

        # 安排下一个到达事件
        next_arrival_time = self.curr_time + random.expovariate(self.lambd)
        self.schedule_event(Event(next_arrival_time, 'arrival', self.next_pkt_id))
        self.next_pkt_id += 1

        # 当前包加入排队
        self.queue.append((event.pkt_id, self.curr_time))

        # 若无服务中包，则启动服务
        if not self.in_service:
            self.start_service()

    def handle_departure(self, event):
        # 弹出已完成服务的包
        pkt_id, arrival_time = self.queue.pop(0)
        delay = self.curr_time - arrival_time
        print(f"[{event.time:.2f}]: pkt {pkt_id} departs having spent {delay:.2f} us in the system")

        # 累计统计数据
        self.total_delay += delay
        self.total_pkts += 1
        self.pkts_in_system -= 1

        # 若队列中还有包，继续服务下一个
        if self.queue:
            self.start_service()
        else:
            self.in_service = False

    def start_service(self):
        self.in_service = True
        pkt_id, _ = self.queue[0]

        # 包大小服从指数分布（mean = MEAN_PKT_SIZE_BITS）
        pkt_size_bits = random.expovariate(1 / MEAN_PKT_SIZE_BITS)

        # 服务时间 = 包大小 / 链路速率，换成微秒
        service_time = (pkt_size_bits / LINK_RATE_BPS) * 1e6
        departure_time = self.curr_time + service_time

        # 安排离开事件
        self.schedule_event(Event(departure_time, 'departure', pkt_id))

    def print_stats(self):
        # 计算平均系统长度 N 和平均时延 T
        N = self.area_under_q / self.curr_time
        T = self.total_delay / self.total_pkts

        print(f"\nSimulation Results (λ = {self.lambd}):")
        print(f"Total packets simulated (departed): {self.total_pkts}")
        print(f"Total packets arrived: {self.next_pkt_id}")
        print(f"Average number in system (N): {N:.4f}")
        print(f"Average time in system (T): {T:.4f} us")

        # 输出 P(n) 分布
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

        # 检查总概率是否接近 1
        total_prob = sum(pn_probs)
        print(f"[Check] Total probability = {total_prob:.6f}")

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

# ========== 主程序入口 ==========
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mm1_simulator.py <npkts> <lambda>")
        sys.exit(1)

    npkts = int(sys.argv[1])       # 包数
    lambd = float(sys.argv[2])     # λ 到达率（单位：每微秒）

    sim = MM1Simulator(npkts, lambd)
    sim.run()
    sim.print_stats()
