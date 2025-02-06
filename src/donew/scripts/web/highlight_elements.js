(() => {
    // Remove existing highlights
    document.querySelectorAll('.DoSee-highlight').forEach(el => el.remove());

    // Create new highlights
    const elements = window.DoSeeElements || {};
    for (const [id, metadata] of Object.entries(elements)) {
        if (!metadata.bounding_box || !metadata.state.isVisible) continue;



        if (!metadata.is_interactive && !(metadata.element_type in ['button', 'link', 'input'])) continue;
            
        

        const box = metadata.bounding_box;
        const highlight = document.createElement('div');
        highlight.className = 'DoSee-highlight';


        // Set color based on element type
        const colors = {
            button: '#FF6B6B',
            link: '#4ECDC4',
            input: '#45B7D1',
            icon: '#96CEB4',
            text: '#FFEEAD'
        };
        const color = colors[metadata.element_type] || colors.text;


        highlight.style.cssText = `
            left: ${box.x}px;
            top: ${box.y}px;
            width: ${box.width}px;
            height: ${box.height}px;
            border: 1px dotted  ${color};
            --highlight-color: ${color};
        `;

        // Add just the element ID as label
        const label = document.createElement('div');
        label.className = 'DoSee-label';
        label.textContent = id;
        highlight.appendChild(label);

        document.body.appendChild(highlight);
    }
})(); 