const canvas = document.getElementById("littleManCanvas");
const ctx = canvas.getContext("2d");

// Image for the little man
const littleManImg = new Image();
littleManImg.src = "assets/images/littleMan.png"; // 替换为你的小人图片路径

// Little man configuration
const littleMan = {
  x: canvas.width / 2,
  y: canvas.height / 2,
  eyeRadius: 10,
  eyeOffsetX: 70, // 调整为适应720x720图片的眼睛位置
  eyeOffsetY: 40,
  colors: {
    eye: "#000",
    eyeHighlight: "#fff",
  },
};

// Mouse tracking
const mouse = { x: canvas.width / 2, y: canvas.height / 2 };

canvas.addEventListener("mousemove", (e) => {
  mouse.x = e.clientX - canvas.offsetLeft;
  mouse.y = e.clientY - canvas.offsetTop;
});

// Function to resize canvas
function resizeCanvas() {
  canvas.width = window.innerWidth * 0.7;
  canvas.height = window.innerHeight;

  // Update little man's position based on new canvas size
  littleMan.x = canvas.width / 2;
  littleMan.y = canvas.height / 2;

  drawLittleMan(); // Redraw content after resize
}

// Draw the little man and eyes
function drawLittleMan() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw the image
  const imgSize = 720; // Image size (assumes square image)
  const imgX = littleMan.x - imgSize / 2;
  const imgY = littleMan.y - imgSize / 2;

  ctx.drawImage(littleManImg, imgX, imgY, imgSize, imgSize);

  // Calculate eye movement
  const dx = mouse.x - littleMan.x;
  const dy = mouse.y - littleMan.y;
  const distance = Math.sqrt(dx * dx + dy * dy);
  const maxEyeMove = 15; // Maximum eye movement distance
  const eyeMoveX = (dx / distance) * maxEyeMove || 0;
  const eyeMoveY = (dy / distance) * maxEyeMove || 0;

  // Draw eyes
  [1, -1].forEach((side) => {
    ctx.fillStyle = littleMan.colors.eye;
    const eyeX = littleMan.x + side * littleMan.eyeOffsetX + eyeMoveX;
    const eyeY = littleMan.y + littleMan.eyeOffsetY + eyeMoveY;
    ctx.beginPath();
    ctx.arc(eyeX, eyeY, littleMan.eyeRadius, 0, Math.PI * 2);
    ctx.fill();

    // Draw eye highlight
    ctx.fillStyle = littleMan.colors.eyeHighlight;
    ctx.beginPath();
    ctx.arc(eyeX - 3, eyeY - 3, 3, 0, Math.PI * 2);
    ctx.fill();
  });
}

// Handle browser resize event
window.addEventListener("resize", resizeCanvas);

// Start animation
function animate() {
  drawLittleMan();
  requestAnimationFrame(animate);
}

// Wait for the image to load before starting
littleManImg.onload = () => {
  resizeCanvas(); // Ensure canvas size matches window on load
  animate();
};
