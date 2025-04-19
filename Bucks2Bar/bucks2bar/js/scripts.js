// Function to retrieve income and expenses values for each month
function getMonthlyData() {
    const months = [
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december'
    ];

    return months.map(month => {
        const income = parseFloat(document.getElementById(`${month}-income`).value) || 0;
        const expenses = parseFloat(document.getElementById(`${month}-expenses`).value) || 0;
        return { month, income, expenses };
    });
}

document.addEventListener("DOMContentLoaded", function () {
    const ctx = document.getElementById('barChart').getContext('2d');

    // Initialize the bar chart
    const barChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
            datasets: [{
                label: 'Income',
                data: Array(12).fill(0),
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }, {
                label: 'Expenses',
                data: Array(12).fill(0),
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Monthly Income vs Expenses'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // Update chart data when the "Chart" tab is clicked
    const chartTab = document.getElementById('chart-tab');
    chartTab.addEventListener('click', () => {
        const monthlyData = getMonthlyData();

        // Update chart data
        barChart.data.labels = monthlyData.map(data => data.month.charAt(0).toUpperCase() + data.month.slice(1));
        barChart.data.datasets[0].data = monthlyData.map(data => data.income);
        barChart.data.datasets[1].data = monthlyData.map(data => data.expenses);

        // Refresh the chart
        barChart.update();
    });

    const downloadBtn = document.getElementById('downloadBtn');
    const canvas = document.getElementById('barChart');

    downloadBtn.addEventListener('click', () => {
        // 將 canvas 轉換為圖片 URL
        const image = canvas.toDataURL('image/png');

        // 建立一個隱藏的 <a> 元素來觸發下載
        const link = document.createElement('a');
        link.href = image;
        link.download = 'chart.png'; // 設定下載的檔案名稱
        link.click();
    });
});

