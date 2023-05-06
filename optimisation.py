def compute_partition_avg(iterator):
    partition_sum = 0
    partition_count = 0
    for row in iterator:
        partition_sum += sum(row)
        partition_count += len(row)
    partition_avg = partition_sum / partition_count
    print("Partition avg: {}".format(partition_avg))
    yield partition_avg