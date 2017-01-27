/* From http://stackoverflow.com/a/6168353/ */  // NOLINT

#ifndef SRC_LOG_LOGGING_H_
#define SRC_LOG_LOGGING_H_

#include <iostream>
#include <sstream>
#include <string>

#define OUT_STREAM std::clog

enum LogLevel {
    logERROR, logWARNING, logPROGRESS, logINFO, logDEBUG
};

namespace logging {
class Logger {
 private:
    std::ostringstream _buffer;
    LogLevel _level;

    std::string to_string(LogLevel level) {
        switch (level) {
            case logERROR:
                return std::string("[ERROR] : ");
                break;
            case logWARNING:
                return std::string("[WARNING] : ");
                break;
            case logPROGRESS:
                return std::string("[PROGRESS] : ");
                break;
            case logINFO:
                return std::string("[INFO] : ");
                break;
            case logDEBUG:
                return std::string("[DEBUG] : ");
                break;
        }
    }

 public:
    explicit Logger(LogLevel _loglevel = logERROR) {
        _level = _loglevel;
    }
    ~Logger() {
        std::string str = _buffer.str();

        // startswith \r
        if (str.at(0) == '\r') {
            str.erase(0, 1);  // remove first char
            OUT_STREAM << '\r' << to_string(_level) << str << std::flush;
        } else {
            OUT_STREAM << to_string(_level) << std::flush << str << std::flush;
        }
    }

    template <typename T>
    Logger& operator<<(const T& value) {
        _buffer << value;
        return *this;
    }
};
}  // namespace logging

extern LogLevel loglevel;

#define logger(level) if (level > loglevel); else logging::Logger(level)  // NOLINT

#endif  // SRC_LOG_LOGGING_H_
