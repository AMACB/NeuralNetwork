#ifndef SRC_TIMING_H_
#define SRC_TIMING_H_

#include <chrono>
#include <string>

typedef std::chrono::steady_clock::time_point time_point;

namespace timing {
time_point now() {
	return std::chrono::steady_clock::now();
}
double diff(const time_point& a, const time_point& b) {
	return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
}
}  // namespace time

#endif  // SRC_TIMING_H_
