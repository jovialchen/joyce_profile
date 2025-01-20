const ballContainer = document.querySelector('.ball-container');
const filterOptions = document.querySelectorAll('.filter');
/*
.ball.border-programming { border: 3px solid #D88393; }
.ball.border-protocol { border: 3px solid #E7AF9C; }
.ball.border-funarea { border: 3px solid #FDF5BF; }
.ball.border-lang { border: 3px solid #7EC9B9; }
.ball.border-tools { border: 3px solid #CFB0D4; }
.ball.border-soft { border: 3px solid #6A0136; }
.ball.fill-advanced { background-color: #a9e5bb; color: #000; }
.ball.fill-intermediate { background-color: #037171; }
.ball.fill-beginner { background-color: #bd9391; }
*/
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

// Generate balls with rendered Markdown tooltips
async function generateBalls() {
  ballContainer.innerHTML = ''; // 清空之前的小球

  // 使用对象按 border 分类
  const borderGroups = {};

  // 按 border 分类小球
  ballsData.forEach(async (ball) => {
    const div = document.createElement('div');
    div.className = `ball border-${ball.border} fill-${ball.fill}`;
    div.textContent = ball.text;

    // Tooltip
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.innerHTML = await fetchMarkdown(ball.markdown); // 渲染后的 HTML
    div.appendChild(tooltip);

    // 按 border 分类
    const borderClass = `border-${ball.border}`;
    if (!borderGroups[borderClass]) {
      borderGroups[borderClass] = document.createElement('div'); // 创建新容器
      borderGroups[borderClass].className = `border-group ${borderClass}`; // 为容器加上分类标识
      ballContainer.appendChild(borderGroups[borderClass]); // 添加容器到 ballContainer
    }

    borderGroups[borderClass].appendChild(div); // 将小球加入对应的容器
  });
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
