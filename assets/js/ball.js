const ballContainer = document.querySelector('.ball-container');
const filterOptions = document.querySelectorAll('.filter');

// Sample data for balls
const ballsData = [
  { text: 'IMS', border: 'funarea', fill: 'advanced', markdown: 'tooltips/ims.md' },
  { text: '5G', border: 'funarea', fill: 'intermediate', markdown: 'tooltips/5g.md' },
  { text: 'Machine Learning', border: 'funarea', fill: 'advanced', markdown: 'tooltips/ml.md' },
  { text: 'MEGACO', border: 'protocol', fill: 'advanced', markdown: 'tooltips/megaco.md' },
    { text: 'Diameter', border: 'protocol', fill: 'intermediate', markdown: 'tooltips/diameter.md' },
  { text: 'SIP', border: 'protocol', fill: 'intermediate', markdown: 'tooltips/SIP.md' },
  { text: 'C/C++', border: 'programming', fill: 'intermediate', markdown: 'tooltips/cpp_c.md' },
  { text: 'Python', border: 'programming', fill: 'intermediate', markdown: 'tooltips/python.md' },
    { text: 'Robot Framework', border: 'programming', fill: 'intermediate', markdown: 'tooltips/rbtfrmwk.md' },
  { text: 'English', border: 'lang', fill: 'beginner', markdown: 'tooltips/english.md' },
  { text: 'Mandarin', border: 'lang', fill: 'beginner', markdown: 'tooltips/mandarin.md' },
  { text: 'Git', border: 'tools', fill: 'beginner', markdown: 'tooltips/git.md' },
    { text: 'JIRA', border: 'tools', fill: 'beginner', markdown: 'tooltips/jira.md' },
  { text: 'Agile', border: 'soft', fill: 'beginner', markdown: 'tooltips/agile.md' },
  { text: 'SAFe', border: 'soft', fill: 'beginner', markdown: 'tooltips/safe.md' },
  { text: 'Project Management', border: 'soft', fill: 'beginner', markdown: 'tooltips/pm.md' },
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
      const title = document.createElement('div');
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




// 默认全部选中
filterOptions.forEach(option => {
  option.checked = true; // 设置默认选中
});

// Filter functionality
filterOptions.forEach(option => {
  option.addEventListener('change', () => {
    const activeFilters = Array.from(filterOptions)
      .filter(opt => opt.checked) // 获取当前选中的过滤器
      .map(opt => opt.getAttribute('data-type'));

    document.querySelectorAll('.ball').forEach(ball => {
      // 如果小球属于任意一个被选中的分类，则显示，否则隐藏
      ball.style.display = activeFilters.some(filter =>
        ball.classList.contains(filter)
      ) ? 'flex' : 'none';
    });
  });
});




// Initialize balls
generateBalls();
