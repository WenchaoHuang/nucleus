/**
 *	Copyright (c) 2025 Wenchao Huang <physhuangwenchao@gmail.com>
 *
 *	Permission is hereby granted, free of charge, to any person obtaining a copy
 *	of this software and associated documentation files (the "Software"), to deal
 *	in the Software without restriction, including without limitation the rights
 *	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *	copies of the Software, and to permit persons to whom the Software is
 *	furnished to do so, subject to the following conditions:
 *
 *	The above copyright notice and this permission notice shall be included in all
 *	copies or substantial portions of the Software.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *	SOFTWARE.
 */
#pragma once

#include "macros.h"
#include <functional>

namespace NS_NAMESPACE
{
	/*****************************************************************************
	********************************    Logger    ********************************
	*****************************************************************************/

	/**
	 *	@brief		Lightweight logger (singleton).
	 */
	class Logger
	{
		NS_NONCOPYABLE(Logger)

	private:

		//!	@brief		Default constructor.
		Logger() = default;

		//!	@brief		Default destructor.
		~Logger() = default;

	public:

		/**
		 *	@brief		Log levels.
		 */
		enum Level { Assert, Error, Warning, Info, Debug };


		/**
		 *	@brief		Type of message callback function.
		 */
		using LogCallback = std::function<void(const char * file_name, int line, const char * func_name, Level level, const char * log_msg)>;

	public:

		/**
		 *	@brief		Return the singleton instance.
		 *	@warning	Be cautious when multiple dynamic libraries call this function (no longer a singleton in that case).
		 */
		NS_API static Logger * instance()
		{
			static Logger s_instance;

			return &s_instance;
		}


		/**
		 *	@brief		Specify a callback function.
		 *	@param[in]	callback - User-defined callback function for receiving log message.
		 *	@note		Pass nullptr will unregister the callback function.
		 */
		void register_callback(LogCallback callback) noexcept { callback_ = callback; }


		/**
		 *	@brief		Transfer log message to the specified callback function (if registered).
		 *	@param[in]	file_name - Which file sends this message.
		 *	@param[in]	line - Which line in the file invokes this function.
		 *	@param[in]	func_name - Which function invokes this function.
		 *	@param[in]	level - Log level of this message.
		 */
		NS_API void log(const char * file_name, int line, const char * func_name, Level level, const char * format, ...);

	private:

		LogCallback		callback_;
	};
}

/*********************************************************************************
********************************    Log Macros    ********************************
*********************************************************************************/

#define NS_INFO_LOG(...)		NS_NAMESPACE::Logger::instance()->log(__FILE__, __LINE__, __FUNCTION__, NS_NAMESPACE::Logger::Info, __VA_ARGS__)
#define NS_DEBUG_LOG(...)		NS_NAMESPACE::Logger::instance()->log(__FILE__, __LINE__, __FUNCTION__, NS_NAMESPACE::Logger::Debug, __VA_ARGS__)
#define NS_ERROR_LOG(...)		NS_NAMESPACE::Logger::instance()->log(__FILE__, __LINE__, __FUNCTION__, NS_NAMESPACE::Logger::Error, __VA_ARGS__)
#define NS_ASSERT_LOG(...)		NS_NAMESPACE::Logger::instance()->log(__FILE__, __LINE__, __FUNCTION__, NS_NAMESPACE::Logger::Assert, __VA_ARGS__)
#define NS_WARNING_LOG(...)		NS_NAMESPACE::Logger::instance()->log(__FILE__, __LINE__, __FUNCTION__, NS_NAMESPACE::Logger::Warning, __VA_ARGS__)

#define NS_INFO_LOG_IF(condition, ...)			if (condition)	NS_INFO_LOG(__VA_ARGS__)
#define NS_ERROR_LOG_IF(condition, ...)			if (condition)	NS_ERROR_LOG(__VA_ARGS__)
#define NS_ASSERT_LOG_IF(condition, ...)		if (condition)	NS_ASSERT_LOG(__VA_ARGS__);		NS_ASSERT(!(condition))
#define NS_WARNING_LOG_IF(condition, ...)		if (condition)	NS_WARNING_LOG(__VA_ARGS__)

#ifdef NS_DEBUG
	#define NS_DEBUG_LOG_IF(condition, ...)		if (condition)	NS_DEBUG_LOG(__VA_ARGS__)
#else
	#define NS_DEBUG_LOG_IF(condition, ...)
#endif
