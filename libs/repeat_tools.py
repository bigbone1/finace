from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

class RepeatRun():
    def __init__(self, max_repeat_num: int = 3, sleep_seconds = 0.5, max_workers = 5):
        self.max_repeat_num = max_repeat_num
        self.sleep_seconds = sleep_seconds
        self.wait_list = []
        self.max_workers = max_workers

    def run(self, func, iter_list: list):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:  # 你可以设置合适的线程数量
            future_to_item = {executor.submit(func, i): i for i in iter_list}
            for future in as_completed(future_to_item):
                sleep(self.sleep_seconds)
                item = future_to_item[future]
                try:
                    future.result()  # 这会等待任务完成并获取结果，如果有异常会被抛出
                    print(f'Successfully processed: {item}')
                except Exception as e:
                    print(f'Error processing {item}: {str(e)}')
                    self.wait_list.append(item)

        # for i in iter_list:
        #     print(f'processing: {i}')
        #     try:
        #         sleep(self.sleep_seconds)
        #         func(i)
        #     except Exception as e:
        #         print(str(e))
        #         self.wait_list[0].append(i)

        
        # for i in enumerate(self.wait_list[:3]):
        #     for j in i[1]:
        #         try:
        #             sleep(self.sleep_seconds)
        #             func(j)
        #         except:
        #             self.wait_list[i[0]+1].append(j)

        # return self.wait_list[-1]

        wait_list = self.wait_list
        # for i in range(self.max_repeat_num):
        #     print(f'repeat num: {i}')
        #     if len(wait_list) != 0:
        #         task = RepeatRun(max_repeat_num=1, max_workers=10)
        #         wait_list = task.run(func, wait_list)
        return wait_list


        
    

