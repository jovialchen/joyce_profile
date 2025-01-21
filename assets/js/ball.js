const ballContainer = document.querySelector('.ball-container');
const filterOptions = document.querySelectorAll('.filter');

// Sample data for balls
const ballsData = [
  { text: 'IMS', border: 'funarea', fill: 'advanced', markdown: 'tooltips/ims.md' },
  { text: '5G', border: 'funarea', fill: 'intermediate', markdown: 'tooltips/ims.md' },
  { text: 'Machine Learning', border: 'funarea', fill: 'advanced', markdown: 'tooltips/ims.md' },
  { text: 'MEGACO', border: 'protocol', fill: 'advanced', markdown: 'tooltips/ims.md' },
    { text: 'Diameter', border: 'protocol', fill: 'intermediate', markdown: 'tooltips/ims.md' },
  { text: 'SIP', border: 'protocol', fill: 'intermediate', markdown: 'tooltips/ims.md' },
  { text: 'C/C++', border: 'programming', fill: 'intermediate', markdown: 'tooltips/ims.md' },
  { text: 'Python', border: 'programming', fill: 'intermediate', markdown: 'tooltips/ims.md' },
    { text: 'Robot Framework', border: 'programming', fill: 'intermediate', markdown: 'tooltips/ims.md' },
  { text: 'English', border: 'lang', fill: 'beginner', markdown: 'tooltips/ims.md' },
  { text: 'Mandarin', border: 'lang', fill: 'beginner', markdown: 'tooltips/ims.md' },
  { text: 'Git', border: 'tools', fill: 'beginner', markdown: 'tooltips/ims.md' },
    { text: 'JIRA', border: 'tools', fill: 'beginner', markdown: 'tooltips/ims.md' },
  { text: 'Agile', border: 'soft', fill: 'beginner', markdown: 'tooltips/ims.md' },
  { text: 'SAFe', border: 'soft', fill: 'beginner', markdown: 'tooltips/ims.md' },
  { text: 'Project Management', border: 'soft', fill: 'beginner', markdown: 'tooltips/ims.md' },
];
// Initialize markdown-it
const md = window.markdownit();

// Fetch and render Markdown
async function fetchMarkdown(url) {
  const response = await fetch(url);
  if (!response.ok) return 'Error loading content';

  const markdownText = await response.text();
  return md.render(markdownText); // Render Markdown to HTML
}

async function generateBalls() {
  ballContainer.innerHTML = ''; // 清空之前的小球

  // 使用对象按 border 分类
  const borderGroups = {};
  const borderTitles = {
    'funarea': 'Function Areas',
    'programming': 'Programming Languages',
    'protocol': 'Telecom Protocol Stack',
    'lang': 'Languages',
    'tools': 'Tools',
    'soft': 'Soft Skills'
  };

  // 按 border 分类小球
  for (const ball of ballsData) {
    const borderClass = ball.border;

    // 如果这个分类的容器还不存在，则创建
    if (!borderGroups[borderClass]) {
      const groupContainer = document.createElement('div');
      groupContainer.className = `border-group ${borderClass}`;

      // 添加标题
      const title = document.createElement('h2');
      title.className = 'group-title';
      title.textContent = borderTitles[borderClass];
      groupContainer.appendChild(title);

      borderGroups[borderClass] = groupContainer;
      ballContainer.appendChild(groupContainer);
    }

    // 创建小球元素
    const div = document.createElement('div');
    div.className = `ball ${ball.fill}`;
    div.textContent = ball.text;

    // Tooltip
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.innerHTML = await fetchMarkdown(ball.markdown); // 渲染后的 HTML
    div.appendChild(tooltip);

    // 将小球加入对应的容器
    borderGroups[borderClass].appendChild(div);
  }
}




// Filter functionality
filterOptions.forEach(option => {
  option.addEventListener('change', () => {
    const activeFilters = Array.from(filterOptions)
      .filter(opt => opt.checked)
      .map(opt => opt.getAttribute('data-type'));

    document.querySelectorAll('.ball').forEach(ball => {
      ball.style.display = activeFilters.every(filter =>
        ball.classList.contains(filter)
      ) || activeFilters.length === 0 ? 'flex' : 'none';
    });
  });
});

// Initialize balls
generateBalls();
