document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const clearButton = document.getElementById('clearButton');
    const predictedDigitElement = document.getElementById('predictedDigit');
    const confidenceElement = document.getElementById('confidence');
    const probabilityBarsElement = document.getElementById('probabilityBars');
    
    // Настройка canvas
    canvas.width = 280;
    canvas.height = 280;
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Создаем элементы для вероятностей
    for (let i = 0; i < 10; i++) {
        const barContainer = document.createElement('div');
        barContainer.className = 'prob-bar';
        
        const label = document.createElement('div');
        label.className = 'prob-label';
        label.innerHTML = `<span>${i}</span><span class="prob-value">0%</span>`;
        
        const barBg = document.createElement('div');
        barBg.className = 'prob-bar-bg';
        
        const barFill = document.createElement('div');
        barFill.className = 'prob-bar-fill';
        barFill.dataset.digit = i;
        
        barBg.appendChild(barFill);
        barContainer.appendChild(label);
        barContainer.appendChild(barBg);
        probabilityBarsElement.appendChild(barContainer);
    }
    
    const probValueElements = document.querySelectorAll('.prob-value');
    
    // Переменные для рисования
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let predictionTimeout;
    
    // Функции для рисования
    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = getCanvasCoords(e);
    }
    
    function draw(e) {
        if (!isDrawing) return;
        
        const [x, y] = getCanvasCoords(e);
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        
        [lastX, lastY] = [x, y];
        
        // Запускаем предсказание с задержкой
        clearTimeout(predictionTimeout);
        predictionTimeout = setTimeout(predictDigit, 200);
    }
    
    function stopDrawing() {
        isDrawing = false;
        predictDigit();
    }
    
    function getCanvasCoords(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        
        const clientX = e.clientX || e.touches[0].clientX;
        const clientY = e.clientY || e.touches[0].clientY;
        
        return [
            (clientX - rect.left) * scaleX,
            (clientY - rect.top) * scaleY
        ];
    }
    
    // Функция предсказания цифры
    async function predictDigit() {
        const imageData = canvas.toDataURL('image/png');
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });
            
            const result = await response.json();
            
            // Обновляем интерфейс
            predictedDigitElement.textContent = result.predicted;
            const confidence = (result.probabilities[result.predicted] * 100).toFixed(1);
            confidenceElement.textContent = `${confidence}%`;
            
            // Обновляем вероятности
            result.probabilities.forEach((prob, i) => {
                const percentage = (prob * 100).toFixed(1);
                probValueElements[i].textContent = `${percentage}%`;
                
                const barFill = document.querySelector(`.prob-bar-fill[data-digit="${i}"]`);
                barFill.style.width = `${percentage}%`;
                
                // Подсвечиваем максимальную вероятность
                if (i === result.predicted) {
                    barFill.style.backgroundColor = '#e74c3c';
                } else {
                    barFill.style.backgroundColor = '#2ecc71';
                }
            });
            
        } catch (error) {
            console.error('Prediction error:', error);
        }
    }
    
    // Очистка canvas
    function clearCanvas() {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        predictedDigitElement.textContent = '-';
        confidenceElement.textContent = '0%';
        
        document.querySelectorAll('.prob-bar-fill').forEach(bar => {
            bar.style.width = '0%';
            bar.style.backgroundColor = '#2ecc71';
        });
        
        document.querySelectorAll('.prob-value').forEach(el => {
            el.textContent = '0%';
        });
    }
    
    // Обработчики событий
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        startDrawing(e.touches[0]);
    });
    
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        draw(e.touches[0]);
    });
    
    canvas.addEventListener('touchend', stopDrawing);
    
    clearButton.addEventListener('click', clearCanvas);
    
    // Инициализация
    clearCanvas();
});