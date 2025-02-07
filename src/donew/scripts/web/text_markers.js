(() => {
    const elements = window.DoSeeElements || {};
    const placeholders = [];

    /**
     * Extract text content from an element using the following priority:
     * 1. Direct text content (if not just whitespace)
     * 2. aria-label on the element itself
     * 3. alt text from child images
     * 4. title attribute
     * 5. aria-labels from child elements
     * 6. value attribute (for inputs/buttons)
     * 7. placeholder attribute (for inputs)
     * 8. selected option text (for selects)
     * 9. Combination of the above for complex elements
     * 
     * Common cases handled:
     * - <a href="...">Simple text</a>
     * - <a href="..."><img alt="Logo"></a>
     * - <a href="..." aria-label="Search"><i class="fa fa-search"></i></a>
     * - <a href="..."><svg aria-label="GitHub"></svg></a>
     * - <button><i class="fa fa-search"></i> Search</button>
     * - <button aria-label="Close"><svg>...</svg></button>
     * - <input type="submit" value="Submit">
     * - <select><option value="1">Option 1</option></select>
     * - <button><img alt="Save"> Save Changes</button>
     */
    function extractElementText(elem) {
        let texts = [];

        // 1. Check direct text content (excluding child element texts)
        const directText = Array.from(elem.childNodes)
            .filter(node => node.nodeType === Node.TEXT_NODE)
            .map(node => node.textContent.trim())
            .filter(text => text.length > 0);
        texts.push(...directText);

        // 2. Check aria-label on the element itself
        const ariaLabel = elem.getAttribute('aria-label');
        if (ariaLabel) texts.push(ariaLabel);

        // 3. Check child images for alt text
        const images = elem.getElementsByTagName('img');
        for (const img of images) {
            const alt = img.getAttribute('alt');
            if (alt) texts.push(alt);
        }

        // 4. Check title attribute
        const title = elem.getAttribute('title');
        if (title) texts.push(title);

        // 5. Check child elements for aria-labels and text
        const children = elem.children;
        for (const child of children) {
            const childAriaLabel = child.getAttribute('aria-label');
            if (childAriaLabel) texts.push(childAriaLabel);

            // For SVGs, check both aria-label and nested text
            if (child.tagName.toLowerCase() === 'svg') {
                const svgText = child.textContent.trim();
                if (svgText) texts.push(svgText);
            }
        }

        // 6. Special handling for form elements
        const tagName = elem.tagName.toLowerCase();
        if (tagName === 'input' || tagName === 'button') {
            // Check value attribute
            const value = elem.getAttribute('value');
            if (value) texts.push(value);

            // Check placeholder for inputs
            if (tagName === 'input') {
                const placeholder = elem.getAttribute('placeholder');
                if (placeholder) texts.push(placeholder);
            }
        } else if (tagName === 'select') {
            // Get selected option text
            const selectedOption = elem.querySelector('option[selected]') || elem.options[elem.selectedIndex];
            if (selectedOption) {
                const optionText = selectedOption.textContent.trim();
                if (optionText) texts.push(optionText);
            }
            // Also add placeholder-like first option if no selection
            if (!elem.value && elem.options.length > 0) {
                const firstOptionText = elem.options[0].textContent.trim();
                if (firstOptionText) texts.push(`(${firstOptionText})`);
            }
        }

        // Combine all found texts and remove duplicates
        return [...new Set(texts)].join(' ');
    }

    // Store original display values and tag interactive elements
    for (const [id, metadata] of Object.entries(elements)) {
        if (metadata.is_interactive) {
            const elem = document.querySelector(`[data-dosee-element-id='${id}']`);
            if (elem && window.getComputedStyle(elem).display !== 'none') {  // only process visible elements
                if (['INPUT', 'TEXTAREA'].includes(elem.tagName)) {
                    // Store the original value for restoration
                    elem.dataset.originalValue = elem.value;
                    let elementText = elem.value.trim();
                    if (!elementText) {
                        elementText = extractElementText(elem);
                    }
                    let marker = `@${id}${elementText ? ' - ' + elementText : ''}`;
                    // Set the element's value to the marker
                    elem.value = marker;
                } else {
                    // Store the original content for later restoration
                    elem.dataset.originalContent = elem.innerHTML;
                    let elementText = elem.innerText.trim().replace(/\s+/g, ' ');
                    if (!elementText) {
                        elementText = extractElementText(elem);
                    }
                    let marker = `@${id}${elementText ? ' - ' + elementText : ''}`;
                    // Replace the element's text content with the marker
                    elem.textContent = marker;
                }
            }
        }
    }
    return placeholders.length;
})(); 