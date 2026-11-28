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

#include <string>
#include <chrono>
#include <thread>
#include <vector>
#include <Windows.h>


struct alignas(4) ColorRGB
{
	unsigned char R, G, B, A;
};


class ExampleWindow
{

public:

	ExampleWindow() : hwnd_(NULL), width_(0), height_(0) {}

	explicit ExampleWindow(const char * title, int width, int height) : ExampleWindow()
	{
		hwnd_ = this->create_window(title, width, height);

		image_.assign(width * height, ColorRGB{ 255, 255, 255, 255 });

		height_ = height;

		width_ = width;
	}

public:

	bool closed() const { return hwnd_ == NULL; }

	void show() { ShowWindow(hwnd_, SW_SHOWNORMAL); }

	void processEvents()
	{
		if (hwnd_)
		{
			MSG message = {};

			while (::PeekMessage(&message, hwnd_, 0, 0, PM_REMOVE))
			{
				::TranslateMessage(&message);

				::DispatchMessage(&message);
			};
		}
	}

	void updateImage(const std::vector<ColorRGB> & image)
	{
		image_ = image;
		UpdateWindow(hwnd_);
		InvalidateRect(hwnd_, NULL, FALSE);
	}

	void setTitle(const std::string & text)
	{
		::SetWindowText(hwnd_, text.c_str());
	}

private:

	HWND create_window(const char * title, int w, int h)
	{
		WNDCLASSEX wc = {};
		wc.cbSize = sizeof(WNDCLASSEX);
		wc.style = CS_HREDRAW | CS_VREDRAW;
		wc.lpfnWndProc = ExampleWindow::WndProc;
		wc.cbClsExtra = 0;
		wc.cbWndExtra = 0;
		wc.hInstance = GetModuleHandle(NULL);
		wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
		wc.hCursor = NULL;
		wc.hbrBackground = (HBRUSH)GetStockObject(NULL_BRUSH);
		wc.lpszMenuName = NULL;
		wc.lpszClassName = "Julia Set CUDA";
		wc.hIconSm = LoadIcon(NULL, IDI_WINLOGO);
		RegisterClassEx(&wc);

		DWORD dwStyle = WS_OVERLAPPEDWINDOW;
		DWORD dwExStyle = WS_EX_APPWINDOW;

		RECT rect = { 0L, 0L, w, h };
		AdjustWindowRectEx(&rect, dwStyle, FALSE, dwExStyle);
		h = rect.bottom - rect.top;
		w = rect.right - rect.left;

		return CreateWindowEx(dwExStyle, wc.lpszClassName, title, dwStyle, CW_USEDEFAULT, CW_USEDEFAULT, w, h, NULL, NULL, wc.hInstance, this);
	}

	static LRESULT WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
	{
		ExampleWindow * window = nullptr;

		if (msg == WM_CREATE)		//!	Retrieve the use pointer.
		{
			CREATESTRUCT * pCreate = reinterpret_cast<CREATESTRUCT*>(lParam);

			window = reinterpret_cast<ExampleWindow*>(pCreate->lpCreateParams);

			::SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)window);
		}
		else
		{
			LONG_PTR ptr = ::GetWindowLongPtr(hwnd, GWLP_USERDATA);

			window = reinterpret_cast<ExampleWindow*>(ptr);
		}

		switch (msg)
		{	
			case WM_PAINT:
			{
				if (window)
				{
					PAINTSTRUCT ps;
					HDC hdc = BeginPaint(hwnd, &ps);

					BITMAPINFO bmi = {};
					bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
					bmi.bmiHeader.biWidth = window->width_;
					bmi.bmiHeader.biHeight = -window->height_;
					bmi.bmiHeader.biPlanes = 1;
					bmi.bmiHeader.biBitCount = 32;
					bmi.bmiHeader.biCompression = BI_RGB;

					StretchDIBits(hdc,
								  0, 0, window->width_, window->height_,
								  0, 0, window->width_, window->height_,
								  window->image_.data(), &bmi, DIB_RGB_COLORS, SRCCOPY);

					EndPaint(hwnd, &ps);
				}

				break;
			}
			case WM_CLOSE:
				window->hwnd_ = NULL;
				break;
			case WM_DESTROY:
				PostQuitMessage(0);
				return 0;
		}

		return DefWindowProc(hwnd, msg, wParam, lParam);
	}

private:

	HWND						hwnd_;
	int							width_;
	int							height_;
	std::vector<ColorRGB>		image_;
};