// Markdown-It instance
const md = window.markdownit();

// Markdown files directory
const markdownDir = 'markdowns/';

// List of Markdown files
const markdownFiles = ['file1.md', 'file2.md', 'file3.md'];

// Dynamically generate navigation links
const navLinks = document.getElementById('nav-links');
markdownFiles.forEach((file) => {
  const fileName = file.replace('.md', ''); // Remove .md extension
  const link = document.createElement('a');
  link.href = `?file=${file}`;
  link.textContent = fileName; // Display file name as link text
  navLinks.appendChild(link);
});

// Fetch Markdown file from URL parameter
const params = new URLSearchParams(window.location.search);
const fileName = params.get('file') || markdownFiles[0]; // Default to the first file

// Fetch and render the Markdown content
fetch(`${markdownDir}${fileName}`)
  .then((response) => {
    if (!response.ok) {
      throw new Error(`Cannot load file: ${fileName}`);
    }
    return response.text();
  })
  .then((markdown) => {
    // Render the Markdown to HTML
    const renderedContent = md.render(markdown);
    document.getElementById('rendered-content').innerHTML = renderedContent;
  })
  .catch((error) => {
    document.getElementById('rendered-content').innerHTML = `<p style="color: red;">${error.message}</p>`;
  });

