import time
from datetime import datetime
import pytz

class EstimateTime:
    def __init__(self, epoch, start_time) -> None:
        self.epoch = epoch
        self.now_time = time.time()
        self.last_time = start_time

        self.london_timezone = pytz.timezone('Europe/London')

    def estimate(self, now_epoch):
        self.now_time = time.time()
        time_of_last_epoch = self.now_time - self.last_time
        remain_epoch = self.epoch - now_epoch
        remain_time = time_of_last_epoch * remain_epoch
        finishing_time = self.now_time + remain_time

        self.last_time = self.now_time

        return datetime.fromtimestamp(finishing_time, tz=self.london_timezone).strftime("%Y-%m-%d %H:%M:%S")
    
if __name__ == '__main__':
    time_estimate = EstimateTime(epoch=100, start_time=time.time())

    for now_epoch in range(100):
        time_str = time_estimate.estimate(now_epoch=now_epoch)
        print(time_str)

        time.sleep(5)