
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>

#include <asm/unistd.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>

#include "grid.h"

static int perf_open_counter(int group_fd, uint64_t config, uint64_t config1)
{
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.size = sizeof(attr);
    attr.type = PERF_TYPE_RAW;
    attr.config = config;
    attr.config1 = config1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.disabled = 1;
    attr.inherit = 1;
    attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return syscall(__NR_perf_event_open, &attr, 0, -1, group_fd, 0);
}

void open_perf_counters(struct perf_counters *counter)
{

    counter->x87 = perf_open_counter(-1, 0x530110, 0);
    counter->sse_sd = perf_open_counter(-1, 0x538010, 0);
    counter->sse_pd = perf_open_counter(-1, 0x531010, 0);
    counter->llc_miss = perf_open_counter(-1, 0x5301b7,0x3f80400091UL);

}

void enable_perf_counters(struct perf_counters *counter)
{

    ioctl(counter->x87, PERF_EVENT_IOC_ENABLE);
    ioctl(counter->sse_sd, PERF_EVENT_IOC_ENABLE);
    ioctl(counter->sse_pd, PERF_EVENT_IOC_ENABLE);
    ioctl(counter->llc_miss, PERF_EVENT_IOC_ENABLE);

}

void disable_perf_counters(struct perf_counters *counter)
{

    ioctl(counter->x87, PERF_EVENT_IOC_DISABLE);
    ioctl(counter->sse_sd, PERF_EVENT_IOC_DISABLE);
    ioctl(counter->sse_pd, PERF_EVENT_IOC_DISABLE);
    ioctl(counter->llc_miss, PERF_EVENT_IOC_DISABLE);

}

static uint64_t read_perf_counter(int counter_fd)
{
    uint64_t values[3] = { 0, 0 };
    read(counter_fd, &values, sizeof(values));
    return values[0];
}

static uint64_t read_perf_counter_enabled(int counter_fd)
{
    uint64_t values[3] = { 0, 0 };
    read(counter_fd, &values, sizeof(values));
    return values[1];
}

static uint64_t read_perf_counter_running(int counter_fd)
{
    uint64_t values[3] = { 0, 0 };
    read(counter_fd, &values, sizeof(values));
    return values[2];
}

void print_perf_counters(struct perf_counters *counter,
                         uint64_t expected_flops,
                         uint64_t expected_mem)
{

    const double giga = 1000000000;
    const double cache_line = 64;

    double running = read_perf_counter_running(counter->x87);
    double enabled = read_perf_counter_enabled(counter->x87);
    if (running != enabled) {
        printf("Performance counters not running or multiplexed! Results might be wrong...\n");
    }

    double x87     = read_perf_counter(counter->x87);
    double sse_sd  = read_perf_counter(counter->sse_sd);
    double sse_pd  = read_perf_counter(counter->sse_pd);
    double llc_miss= read_perf_counter(counter->llc_miss);
    double flops = x87 + sse_sd + 2*sse_pd;

    printf("X87:                %.2f Gop\n", x87 / giga);
    printf("SSE scalar double:  %.2f Gop\n", sse_sd / giga);
    printf("SSE packed double:  %.2f Gop\n", sse_pd / giga);
    printf("LLC miss:           %.2f GB (expected %.2f GB)\n\n", cache_line * llc_miss / giga, expected_mem / giga);
    printf("Total FLOPS:        %.2f Gop (expected %.2f Gop)\n", flops / giga, expected_flops / giga);
    printf("Rate:               %.2f Gop/s (expected %.2f op/s)\n", flops / running, (double)expected_flops / running);
    printf("Intensity:          %.4f op/B (expected %.2f op/B)\n", flops / (cache_line * llc_miss), (double)expected_flops / expected_mem);

}
