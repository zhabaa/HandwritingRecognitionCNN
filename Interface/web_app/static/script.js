document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const clearButton = document.getElementById('clearButton');

    canvas.width = 64;
    canvas.height = 64;
    
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    function getCanvasCoords(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    function startDrawing(e) {
        isDrawing = true;
        const coords = getCanvasCoords(e);
        [lastX, lastY] = [coords.x, coords.y];
        
        ctx.beginPath();
        ctx.arc(lastX, lastY, ctx.lineWidth/2, 0, Math.PI*2);
        ctx.fill();
    }

    function draw(e) {
        if (!isDrawing) return;
        
        const coords = getCanvasCoords(e);
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();
        
        [lastX, lastY] = [coords.x, coords.y];
    }

    function stopDrawing() {
        isDrawing = false;
    }

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    clearButton.addEventListener('click', function() {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    });

    function handleTouchStart(e) {
        e.preventDefault();
        startDrawing(e.touches[0]);
    }
    
    function handleTouchMove(e) {
        e.preventDefault();
        draw(e.touches[0]);
    }

    canvas.addEventListener('touchstart', handleTouchStart);
    canvas.addEventListener('touchmove', handleTouchMove);
    canvas.addEventListener('touchend', stopDrawing);

    document.getElementById('saveButton').addEventListener('click', function() {
        const link = document.createElement('a');
        link.download = 'drawing-' + new Date().toISOString().slice(0, 10) + '.png';
        
        canvas.toBlob(function(blob) {
            link.href = URL.createObjectURL(blob);
            
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            setTimeout(() => URL.revokeObjectURL(link.href), 100);
        }, 'image/png');
    });
});