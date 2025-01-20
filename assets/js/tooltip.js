// assets/js/section3.js
document.addEventListener('DOMContentLoaded', () => {
    fetch('sections/section3.html')
        .then(response => response.text())
        .then(html => {
            document.getElementById('section3').innerHTML = html;
            loadTooltips();
        });

    function loadTooltips() {
        const skills = document.querySelectorAll('.skill');
        const md = window.markdownit();

        skills.forEach(skill => {
            const tooltipContent = skill.dataset.tooltip; // Get Markdown filename
            if (tooltipContent) {
                fetch(`tooltips/${tooltipContent}.md`) // Fetch the Markdown file
                    .then(response => response.text())
                    .then(markdown => {
                        const tooltip = skill.querySelector('.tooltip');
                        if (tooltip) {
                            tooltip.innerHTML = `<div class="markdown-body">${md.render(markdown)}</div>`; // Render Markdown
                        }
                    }).catch(error=>{
                        const tooltip = skill.querySelector('.tooltip');
                        if (tooltip) {
                            tooltip.innerHTML = `<div class="markdown-body"><p>Tooltip content not found</p></div>`; // Render Markdown
                        }
                    });
            }
        });
    }
});