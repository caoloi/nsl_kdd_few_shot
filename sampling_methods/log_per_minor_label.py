from records import TRAIN_RECORDS, TEST_RECORDS
import numpy as np

N = 117.8
# records = TRAIN_RECORDS
records = TEST_RECORDS

log_sum = 0
for major_label in records:
  for minor_label in records[major_label]:
    log_sum += np.log(records[major_label][minor_label] + 1)

print(log_sum)

total_count = 0
for major_label in records:
  for minor_label in records[major_label]:
    count = int(N * np.log(records[major_label][minor_label] + 1) / log_sum)
    print(
        # major_label,
        # minor_label,
        count,
    )
    total_count += count

print(total_count)
