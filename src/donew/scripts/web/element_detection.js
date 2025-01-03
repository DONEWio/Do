(() => {
    function generateXPath(element) {
        // If element has an ID that's not "undefined", use it
        if (element.id && element.id !== "undefined") {
            return `xpath=//*[@id="${element.id}"]`;
        }

        // Generate path without relying on ID
        const path = [];
        let current = element;

        while (current && current.nodeType === Node.ELEMENT_NODE) {
            let selector = current.tagName.toLowerCase();

            // Add index if there are siblings
            const siblings = Array.from(current.parentNode?.children || [])
                .filter(e => e.tagName === current.tagName);

            if (siblings.length > 1) {
                const index = siblings.indexOf(current) + 1;
                selector += `[${index}]`;
            }

            path.unshift(selector);
            current = current.parentNode;
        }

        // Return XPath with xpath= prefix
        return `xpath=//${path.join('/')}`;
    }

    function findElementLabel(element) {
        // Case 1: Check for aria-label attribute
        const ariaLabel = element.getAttribute('aria-label');
        if (ariaLabel) {
            return ariaLabel;
        }

        // Case 2: Check for aria-labelledby
        const labelledBy = element.getAttribute('aria-labelledby');
        if (labelledBy) {
            const labelElement = document.getElementById(labelledBy);
            if (labelElement) {
                return labelElement.textContent.trim();
            }
        }

        // Case 3: Check for associated label using 'for' attribute
        if (element.id) {
            const labelElement = document.querySelector(`label[for="${element.id}"]`);
            if (labelElement) {
                return labelElement.textContent.trim();
            }
        }

        // Case 4: Check if element is wrapped in a label
        let parent = element.parentElement;
        while (parent) {
            if (parent.tagName === 'LABEL') {
                // Get label text excluding the input's own text
                const clone = parent.cloneNode(true);
                const inputs = clone.querySelectorAll('input, select, textarea');
                inputs.forEach(input => input.remove());
                return clone.textContent.trim();
            }
            parent = parent.parentElement;
        }

        // Case 5: Check for placeholder as fallback
        const placeholder = element.getAttribute('placeholder');
        if (placeholder) {
            return placeholder;
        }

        return null;
    }

    function processInteractiveElement(element) {
        // Process buttons and links
        let elementType = 'text';
        if (element.tagName === 'BUTTON' || element.getAttribute('role') === 'button') {
            elementType = 'button';
        } else if (element.tagName === 'A') {
            elementType = 'link';
        }

        return {
            elementType,
            isInteractive: true
        };
    }

    function processFormElement(element) {
        // Process input, select, and textarea elements
        const elementType = 'input';
        const elementLabel = findElementLabel(element);

        return {
            elementType,
            elementLabel,
            isInteractive: true
        };
    }

    function getElementMetadata(element, elementId) {
        const rect = element.getBoundingClientRect();
        const computedStyle = window.getComputedStyle(element);

        // Get attributes
        const attributes = {};
        for (const attr of element.attributes) {
            attributes[attr.name] = attr.value;
        }

        // Process by element type
        let elementInfo = {
            elementType: 'text',
            elementLabel: null,
            isInteractive: false
        };

        if (element.tagName.toUpperCase() === 'SVG') {
            elementInfo = { elementType: 'image', isInteractive: false };
        } else if (['BUTTON', 'A'].includes(element.tagName)) {
            elementInfo = processInteractiveElement(element);
        } else if (['INPUT', 'SELECT', 'TEXTAREA'].includes(element.tagName)) {
            elementInfo = processFormElement(element);
        } else if (element.tagName === 'IMG' || element.getAttribute('role') === 'img') {
            elementInfo = { elementType: 'image', isInteractive: false };
        }

        // Return the same ElementMetadata structure as before
        return {
            element_id: elementId,
            element_name: element.tagName.toLowerCase(),
            element_html: element.outerHTML,
            element_label: elementInfo.elementLabel,
            xpath: generateXPath(element),
            bounding_box: {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height
            },
            is_interactive: elementInfo.isInteractive,
            element_type: elementInfo.elementType,
            attributes: attributes,
            computed_styles: {
                display: computedStyle.display,
                visibility: computedStyle.visibility,
                position: computedStyle.position
            },
            listeners: Object.keys(element).filter(key => key.startsWith('on')),
            parent_id: element.parentElement ? parseInt(element.parentElement.dataset.doseeElementId) : null,
            children_ids: Array.from(element.children)
                .map(child => parseInt(child.dataset.doseeElementId))
                .filter(id => !isNaN(id)),
            state: {
                isVisible: computedStyle.display !== 'none' && computedStyle.visibility !== 'hidden',
                isEnabled: !element.disabled,
                isChecked: element.checked
            }
        };
    }

    function detectElements() {
        const elements = {};
        let elementId = 1;

        function processElement(element) {
            // Skip script and style elements
            if (!element || element.tagName === 'SCRIPT' || element.tagName === 'STYLE') {
                return;
            }

            // Set unique element ID
            element.dataset.doseeElementId = elementId;

            // Get element metadata
            elements[elementId] = getElementMetadata(element, elementId);
            elementId++;

            // Skip processing children for SVG elements
            if (element.tagName.toUpperCase() === 'SVG') {
                return;
            }

            // Process children
            for (const child of element.children) {
                processElement(child);
            }
        }

        // Start processing from documentElement (html) or body if available
        const startElement = document.body || document.documentElement;
        if (startElement) {
            processElement(startElement);
        } else {
            // If no html/body (e.g. raw SVG), process the first child
            const firstElement = document.firstElementChild;
            if (firstElement) {
                processElement(firstElement);
            }
        }

        // Store elements globally for annotation system
        window.DoSeeElements = elements;

        return elements;
    }

    // Execute and return results
    return detectElements();
})(); 