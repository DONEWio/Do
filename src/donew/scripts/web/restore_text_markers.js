(() => {
    const elements = window.DoSeeElements || {};
    const policy = window.trustedTypes ? window.trustedTypes.createPolicy('restorePolicy', { createHTML: input => input }) : null;
    for (const [id, metadata] of Object.entries(elements)) {
        if (metadata.is_interactive) {
            const elem = document.querySelector(`[data-dosee-element-id='${id}']`);
            if (elem && window.getComputedStyle(elem).display !== 'none') {  // only restore visible elements
                if (['INPUT', 'TEXTAREA'].includes(elem.tagName)) {
                    // Restore the original value if it was stored, checking existence instead of truthiness
                    if ('originalValue' in elem.dataset) {
                        elem.value = elem.dataset.originalValue;
                        delete elem.dataset.originalValue;
                    }
                } else {
                    // Restore original content if it was stored
                    if (elem.dataset.originalContent) {
                        elem.innerHTML = policy ? policy.createHTML(elem.dataset.originalContent) : elem.dataset.originalContent;
                        delete elem.dataset.originalContent;
                    }
                }
                // Restore original display
                elem.style.display = '';
            }
        }
    }
})(); 