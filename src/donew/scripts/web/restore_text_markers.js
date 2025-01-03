(() => {
    const elements = window.DoSeeElements || {};
    for (const [id, metadata] of Object.entries(elements)) {
        if (metadata.is_interactive) {
            const elem = document.querySelector(`[data-dosee-element-id='${id}']`);
            if (elem) {
                // Find and remove placeholder
                const prev = elem.previousSibling;
                if (prev && prev.tagName === 'SPAN') {
                    prev.remove();
                }
                // Restore original display
                elem.style.display = '';
            }
        }
    }
})(); 