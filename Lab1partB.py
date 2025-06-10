import sys
import heapq
import pandas as pd
import matplotlib.pyplot as plt

# 固定链路处理速率：10 Gbps
LINK_RATE_BPS = 10_000_000_000

class Event:
    def __init__(self, time, type, pkt_id):
        self.time = time
        self.type = type  # 'arrival' or 'departure'
        self.pkt_id = pkt_id

    def __lt__(self, other):
        return self.time < other.time

class TraceSimulator:
    def __init__(self, trace_file):
        self.trace = pd.read_csv(trace_file, sep=r'\s+', header=None, names=['inter_arrival_us', 'pkt_size_bytes'])
        print(f"Loaded trace with {len(self.trace)} packets.")

        self.queue = []
        self.event_list = []
        self.curr_time = 0.0
        self.in_service = False
        self.total_delay = 0.0
        self.total_pkts = 0
        self.queue_len_record = [0] * 12
        self.pkts_in_system = 0
        self.area_under_q = 0.0
        self.last_event_time = 0.0

    def schedule_event(self, event):
        heapq.heappush(self.event_list, event)

    def run(self):
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

        while self.event_list:
            event = heapq.heappop(self.event_list)
            self.curr_time = event.time
            self.area_under_q += self.pkts_in_system * (self.curr_time - self.last_event_time)
            self.last_event_time = self.curr_time

            if event.type == 'arrival':
                self.handle_arrival(event)
            elif event.type == 'departure':
                self.handle_departure(event)

    def handle_arrival(self, event):
        pkt_size_bytes = self.trace.loc[event.pkt_id, 'pkt_size_bytes']
        service_time_us = (pkt_size_bytes * 8 / LINK_RATE_BPS) * 1e6

        print(f"[{self.curr_time:.2f}]: pkt {event.pkt_id} arrives and finds {len(self.queue)} packets in the queue")

        index = self.pkts_in_system if self.pkts_in_system <= 10 else 11
        self.queue_len_record[index] += 1
        self.pkts_in_system += 1

        self.queue.append((event.pkt_id, self.curr_time, service_time_us))

        if not self.in_service:
            self.start_service()

    def handle_departure(self, event):
        pkt_id, arrival_time, _ = self.queue.pop(0)
        delay = self.curr_time - arrival_time
        print(f"[{self.curr_time:.2f}]: pkt {pkt_id} departs having spent {delay:.2f} us in the system")
        self.total_delay += delay
        self.total_pkts += 1
        self.pkts_in_system -= 1

        if self.queue:
            self.start_service()
        else:
            self.in_service = False

    def start_service(self):
        self.in_service = True
        pkt_id, _, service_time = self.queue[0]
        departure_time = self.curr_time + service_time
        self.schedule_event(Event(departure_time, 'departure', pkt_id))

    def print_stats(self):
        N = self.area_under_q / self.curr_time
        T = self.total_delay / self.total_pkts
        print(f"\nSimulation Results (fixed 10 Gbps system):")
        print(f"Total packets simulated: {self.total_pkts}")
        print(f"Average number in system (N): {N:.4f}")
        print(f"Average time in system (T): {T:.4f} us")

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

        labels = [str(i) for i in range(11)] + ['>10']
        plt.figure(figsize=(10, 6))
        plt.bar(labels, pn_probs)
        plt.xlabel("Number of packets in system (n)")
        plt.ylabel("Probability P(n)")
        plt.title("P(n) Distribution Based on Packet Arrivals")
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Lab01.py <trace_file>")
        sys.exit(1)

    trace_file = sys.argv[1]
    sim = TraceSimulator(trace_file)
    sim.run()
    sim.print_stats()
