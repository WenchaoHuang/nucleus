(() => {
	const menuButton = document.querySelector('[data-menu-toggle]');
	const sidebar = document.querySelector('[data-sidebar]');
	const menuClose = document.querySelector('[data-menu-close]');
	const isMobile = () => window.innerWidth <= 1000;
	const updateMenuVisibility = open => {
		if (sidebar) {
			sidebar.inert = isMobile() && !open;
			if (sidebar.inert) sidebar.setAttribute('aria-hidden', 'true');
			else sidebar.removeAttribute('aria-hidden');
		}
		if (menuClose) menuClose.hidden = !open;
	};
	const closeMenu = (restoreFocus = false) => {
		if (isMobile() && (restoreFocus || sidebar?.contains(document.activeElement))) menuButton?.focus();
		menuButton?.setAttribute('aria-expanded', 'false');
		menuButton?.setAttribute('aria-label', 'Open navigation');
		sidebar?.classList.remove('open');
		document.body.classList.remove('menu-open');
		updateMenuVisibility(false);
	};

	menuButton?.addEventListener('click', () => {
		const open = menuButton.getAttribute('aria-expanded') !== 'true';
		menuButton.setAttribute('aria-expanded', String(open));
		menuButton.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
		sidebar?.classList.toggle('open', open);
		document.body.classList.toggle('menu-open', open);
		updateMenuVisibility(open);
	});

	menuClose?.addEventListener('click', () => closeMenu(true));
	sidebar?.querySelectorAll('a').forEach(link => link.addEventListener('click', () => {
		const wasOpen = menuButton?.getAttribute('aria-expanded') === 'true';
		closeMenu();
		const target = link.hash && link.origin === location.origin && link.pathname === location.pathname ? document.getElementById(link.hash.slice(1)) : null;
		if (wasOpen && target) {
			const temporaryTabIndex = !target.hasAttribute('tabindex');
			if (temporaryTabIndex) target.tabIndex = -1;
			target.focus({ preventScroll: true });
			if (temporaryTabIndex) target.addEventListener('blur', () => target.removeAttribute('tabindex'), { once: true });
		}
	}));
	document.addEventListener('keydown', event => {
		if (menuButton?.getAttribute('aria-expanded') !== 'true') return;
		if (event.key === 'Escape') {
			closeMenu(true);
			return;
		}
		if (event.key === 'Tab') {
			const focusable = [menuButton, ...sidebar.querySelectorAll('a[href], button:not([disabled])')];
			const index = focusable.indexOf(document.activeElement);
			if (index < 0 || (event.shiftKey && index === 0) || (!event.shiftKey && index === focusable.length - 1)) {
				event.preventDefault();
				focusable[event.shiftKey ? focusable.length - 1 : 0].focus();
			}
		}
	});
	document.addEventListener('click', event => {
		if (!sidebar?.contains(event.target) && !menuButton?.contains(event.target)) closeMenu();
	});
	window.addEventListener('resize', () => {
		if (!isMobile()) closeMenu();
		else if (menuButton?.getAttribute('aria-expanded') !== 'true') closeMenu();
	});
	updateMenuVisibility(false);

	document.querySelectorAll('[data-tabs]').forEach(container => {
		const tabs = [...container.querySelectorAll('[role="tab"]')];
		const panels = [...container.querySelectorAll('[data-panel]')];
		const activate = tab => {
			tabs.forEach(item => {
				const selected = item === tab;
				item.setAttribute('aria-selected', String(selected));
				item.tabIndex = selected ? 0 : -1;
			});
			panels.forEach(panel => { panel.hidden = panel.dataset.panel !== tab.dataset.tab; });
		};

		tabs.forEach((tab, index) => {
			tab.addEventListener('click', () => activate(tab));
			tab.addEventListener('keydown', event => {
				if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
				event.preventDefault();
				let target = index;
				if (event.key === 'ArrowLeft') target = (index - 1 + tabs.length) % tabs.length;
				if (event.key === 'ArrowRight') target = (index + 1) % tabs.length;
				if (event.key === 'Home') target = 0;
				if (event.key === 'End') target = tabs.length - 1;
				activate(tabs[target]);
				tabs[target].focus();
			});
		});
		if (tabs.length) activate(tabs.find(tab => tab.getAttribute('aria-selected') === 'true') || tabs[0]);
	});

	const copyStatus = document.querySelector('[data-copy-status]');
	document.querySelectorAll('[data-copy]').forEach(button => {
		let resetTimer;
		button.addEventListener('click', async () => {
			const code = button.closest('.code-block')?.querySelector('pre code');
			const label = button.querySelector('span');
			if (!code) return;
			clearTimeout(resetTimer);
			let message;
			try {
				await navigator.clipboard.writeText(code.textContent);
				message = 'Copied';
				button.classList.add('copied');
			} catch {
				const selection = window.getSelection();
				const range = document.createRange();
				range.selectNodeContents(code);
				selection?.removeAllRanges();
				selection?.addRange(range);
				message = 'Select code';
			}
			if (label) label.textContent = message;
			if (copyStatus) copyStatus.textContent = message === 'Copied' ? 'Code copied to clipboard.' : 'Code selected. Press Ctrl+C or Command+C to copy.';
			resetTimer = setTimeout(() => {
				if (label) label.textContent = 'Copy';
				button.classList.remove('copied');
				if (copyStatus) copyStatus.textContent = '';
			}, 2200);
		});
	});

	const themeButton = document.querySelector('[data-theme-toggle]');
	let savedTheme;
	try { savedTheme = localStorage.getItem('nucleus-theme'); } catch {}
	const applyTheme = theme => {
		const dark = theme === 'dark';
		document.documentElement.dataset.theme = dark ? 'dark' : 'light';
		document.querySelector('meta[name="theme-color"]')?.setAttribute('content', dark ? '#181e1a' : '#f8f9f6');
		themeButton?.setAttribute('aria-pressed', String(dark));
		themeButton?.setAttribute('aria-label', `Switch to ${dark ? 'light' : 'dark'} theme`);
	};
	applyTheme(savedTheme === 'light' ? 'light' : 'dark');
	themeButton?.addEventListener('click', () => {
		savedTheme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
		applyTheme(savedTheme);
		try { localStorage.setItem('nucleus-theme', savedTheme); } catch {}
	});

	const sectionLinks = [...document.querySelectorAll('[data-section-link]')];
	const sections = [...new Set(sectionLinks.map(link => document.getElementById(link.hash.slice(1))).filter(Boolean))];
	let activeSection;
	let selectedSection = location.hash.slice(1);
	let scrollPending = false;
	const updateSection = () => {
		scrollPending = false;
		let active = sections[0];
		let activeTop = -Infinity;
		for (const section of sections) {
			const top = section.getBoundingClientRect().top;
			if (top <= 140 && (top > activeTop + 2 || (Math.abs(top - activeTop) <= 2 && section.id === selectedSection))) {
				active = section;
				activeTop = top;
			}
		}
		if (!active || active.id === activeSection) return;
		activeSection = active.id;
		sectionLinks.forEach(link => {
			const current = link.hash === `#${activeSection}`;
			link.classList.toggle('active', current);
			if (current) link.setAttribute('aria-current', 'location');
			else link.removeAttribute('aria-current');
		});
	};
	sectionLinks.forEach(link => link.addEventListener('click', () => {
		selectedSection = link.hash.slice(1);
	}));
	const scheduleSectionUpdate = () => {
		if (scrollPending) return;
		scrollPending = true;
		window.requestAnimationFrame(updateSection);
	};
	if ('IntersectionObserver' in window) {
		const observer = new IntersectionObserver(scheduleSectionUpdate, { rootMargin: '-100px 0px -60% 0px' });
		sections.forEach(section => observer.observe(section));
	}
	window.addEventListener('scroll', scheduleSectionUpdate, { passive: true });
	window.addEventListener('resize', scheduleSectionUpdate);
	window.addEventListener('hashchange', () => {
		selectedSection = location.hash.slice(1);
		scheduleSectionUpdate();
	});
	updateSection();
})();
