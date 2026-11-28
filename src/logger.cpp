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

#include "logger.h"

#include <mutex>
#include <thread>
#include <cstdio>
#include <string>
#include <cstdarg>
#include <iostream>

#ifdef _WIN32
	#include <windows.h>
#endif

NS_USING_NAMESPACE

/*********************************************************************************
**********************************    Logger    **********************************
*********************************************************************************/

#ifdef _WIN32

static void enable_ansi_colors()
{
	HANDLE handle = ::GetStdHandle(STD_OUTPUT_HANDLE);

	if (handle == INVALID_HANDLE_VALUE)
		return;

	DWORD mode = 0;

	if (!::GetConsoleMode(handle, &mode))
		return;

	::SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}
#endif


void Logger::log(const char * file_name, int line, const char * func_name, Level level, const char * format, ...)
{
#ifdef _WIN32
	static std::once_flag s_ansi_flag;

	std::call_once(s_ansi_flag, enable_ansi_colors);
#endif

	va_list arg_ptr;
	va_list arg_ptr_copy;

	va_start(arg_ptr, format);
	va_copy(arg_ptr_copy, arg_ptr);

	int size = std::vsnprintf(nullptr, 0, format, arg_ptr);

	va_end(arg_ptr);

	if (size < 0)
	{
		va_end(arg_ptr_copy);

		return;
	}

	thread_local std::string log_str;

	log_str.resize(static_cast<size_t>(size) + 1);

	std::vsnprintf(const_cast<char*>(log_str.data()), log_str.size(), format, arg_ptr_copy);

	va_end(arg_ptr_copy);

	//	ANSI color codes
	constexpr const char * COLOR_RED = "\033[31m";
	constexpr const char * COLOR_RESET = "\033[0m";
	constexpr const char * COLOR_GREEN = "\033[32m";
	constexpr const char * COLOR_MAGENTA = "\033[35m";
	constexpr const char * COLOR_ORANGE = "\033[38;2;255;165;0m";
#ifdef NS_DEBUG
	constexpr const char * COLOR_CYAN = "\033[36m";
#endif

	if (callback_ != nullptr)
	{
		callback_(file_name, line, func_name, level, log_str.c_str());
	}
	else if (level == Level::Assert)
	{
		std::cout << COLOR_MAGENTA << " [" << std::this_thread::get_id() << "] Assert: " << func_name << "() => " << log_str << COLOR_RESET << std::endl;
	}
	else if (level == Level::Error)
	{
		std::cout << COLOR_RED << " [" << std::this_thread::get_id() << "] Error: " << func_name << "() => " << log_str << COLOR_RESET << std::endl;
	}
	else if (level == Level::Warning)
	{
		std::cout << COLOR_ORANGE << " [" << std::this_thread::get_id() << "] Warning: " << func_name << "() => " << log_str << COLOR_RESET << std::endl;
	}
	else if (level == Level::Info)
	{
		std::cout << COLOR_GREEN << " [" << std::this_thread::get_id() << "] Info: " << func_name << "() => " << log_str << COLOR_RESET << std::endl;
	}
	else if (level == Level::Debug)
	{
	#ifdef NS_DEBUG
		std::cout << COLOR_CYAN << " [" << std::this_thread::get_id() << "] Debug: " << func_name << "() => " << log_str << COLOR_RESET << std::endl;
	#endif
	}
}
