from records import TRAIN_RECORDS, TEST_RECORDS
import numpy as np

N = 119.2
# records = TRAIN_RECORDS
records = TEST_RECORDS

label_counts = {}
log_counts = {}
log_sum = 0
for major_label in records:
  label_count = 0
  log_count = 0
  for minor_label in records[major_label]:
    label_count += records[major_label][minor_label]
    log_count += np.log(records[major_label][minor_label] + 1)
  label_counts[major_label] = label_count
  log_counts[major_label] = log_count
  log_sum += np.log(label_count)

print(label_counts)
print(log_counts)
print(log_sum)

total_count = 0
for major_label in records:
  label_count = 0
  for minor_label in records[major_label]:
    count = int(
      N
      * (np.log(label_counts[major_label] + 1)) / (log_sum)
      * (np.log(records[major_label][minor_label] + 1)) / (log_counts[major_label])
    )
    print(
        # major_label,
        # minor_label,
        count,
    )
    label_count += count
    total_count += count
  # print("per label", label_count)

print(total_count)
