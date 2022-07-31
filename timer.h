#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <string>
#include <iostream>

/* \return The current time in seconds */
int time_s() {
  using namespace std::chrono;
  return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
};

/* \return The current time in ms */
int time_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
};

/* \return The current time in us */
int time_us() {
  using namespace std::chrono;
  return duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
};

/* \return The current time in ns */
int time_ns() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
};

/* Different units of time */
enum TIME_UNIT {
    s = 0,
    ms,
    us,
    ns
};

/* A class to handle simple timing. */
class Timer {
    public:
        /* \param[in] unit Unit of time to track. */
        Timer(TIME_UNIT unit=TIME_UNIT::ms) {
            unit_ = unit;
            restart();
        };

        /* Restart the timer. */
        void restart() {
            timer_ = get_time_();
        };

        /* Get the current value of the timer. */
        int get() {
            return get_time_() - timer_;
        }

        /* Print the value of the timer.
         * \param[in] message an optional message to show before the timer value */
        void print(std::string message="") {
            int clock = get_time_() - timer_;

            std::cout << message;
            if (message != "") std::cout << " ";

            std::cout << clock << " ";
            switch(unit_) {
                case s:
                    std::cout << "s" << std::endl;
                    break;
                case ms:
                    std::cout << "ms" << std::endl;
                    break;
                case us:
                    std::cout << "us" << std::endl;
                    break;
                case ns:
                    std::cout << "ns" << std::endl;
                    break;
                default:
                    std::cout << std::endl;
                    break;
            }
        }

    private:
        int timer_;
        TIME_UNIT unit_;

        int get_time_() {
            int time = 0;
            switch(unit_) {
                case s:
                    time = time_s();
                    break;
                case ms:
                    time = time_ms();
                    break;
                case us:
                    time = time_us();
                    break;
                case ns:
                    time = time_ns();
                    break;
                default:
                    time = time_ms();
                    break;
            }
            return time;
        };
};

#endif