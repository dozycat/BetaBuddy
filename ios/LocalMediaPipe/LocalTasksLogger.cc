// BetaBuddy SDK customization: use MediaPipe's official no-op logger.
// This replaces only the logger factory, never the pose model or inference graph.
// Headers are pinned to upstream 0ad5a71bcdff3d756dc5b07f93765aaeb4152538.
#define GLOG_EXPORT
#define GLOG_NO_EXPORT
#define GLOG_DEPRECATED [[deprecated]]
#include "mediapipe/tasks/cc/core/logging/factory/logging_factory.h"
#include "mediapipe/tasks/cc/core/logging/tasks_dummy_logger.h"

static_assert(sizeof(mediapipe::Timestamp) == sizeof(int64_t));
namespace mediapipe::tasks::core::logging {
std::unique_ptr<TasksLogger> CreateTasksLogger(const LoggingOptions&) {
  return TasksDummyLogger::Create();
}
}  // namespace mediapipe::tasks::core::logging
