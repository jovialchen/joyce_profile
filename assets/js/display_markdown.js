document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.skill').forEach(skill => {
    const tooltip = skill.querySelector('.tooltip');
    const markdownFile = skill.getAttribute('data-markdown');

    if (markdownFile) {
      fetch(markdownFile)
        .then(response => response.text())
        .then(markdown => {
          const htmlContent = marked.parse(markdown);
          tooltip.innerHTML = htmlContent;
        })
        .catch(error => {
          tooltip.innerHTML = 'Error loading content.';
        });
    } else {
      tooltip.innerHTML = 'No additional information available.';
    }
  });
});
