let currentSection = 0;
const sections = document.querySelectorAll('.section');

function scrollHandler(event) {
  event.preventDefault();
  const delta = event.deltaY;

  if (delta > 0 && currentSection < sections.length - 1) {
    currentSection++;
  } else if (delta < 0 && currentSection > 0) {
    currentSection--;
  }

  sections[currentSection].scrollIntoView({ behavior: 'smooth' });
}
function touchHandler(event) {
  const touch = event.touches[0];
  const delta = touch.clientY - startY;

  if (delta < -50 && currentSection < sections.length - 1) {
    currentSection++;
  } else if (delta > 50 && currentSection > 0) {
    currentSection--;
  }

  sections[currentSection].scrollIntoView({ behavior: 'smooth' });
}

let startY = 0;

window.addEventListener('wheel', scrollHandler, { passive: false });

window.addEventListener('touchstart', (event) => {
  startY = event.touches[0].clientY;
}, { passive: false });

window.addEventListener('touchmove', touchHandler, { passive: false });
window.addEventListener('wheel', scrollHandler, { passive: false });
