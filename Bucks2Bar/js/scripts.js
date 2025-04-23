// Function to retrieve income and expenses values for each month
const getMonthlyData = () => {
    const months = [
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december'
    ];

    return months.map(month => {
        const income = parseFloat(document.getElementById(`${month}-income`)?.value) || 0;
        const expenses = parseFloat(document.getElementById(`${month}-expenses`)?.value) || 0;
        return { month, income, expenses };
    });
};

document.addEventListener("DOMContentLoaded", () => {
    const usernameInput = document.getElementById("username");

    usernameInput?.addEventListener("input", () => { 
        const username = usernameInput.value;
        const regex = /^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&#])[A-Za-z\d@$!%*?&#]{8,}$/;
        const isValid = regex.test(username);

        usernameInput.style.borderColor = isValid ? "green" : "red";
        usernameInput.style.backgroundColor = isValid ? "#d4edda" : "#f8d7da"; // light green or light red
    });

    const ctx = document.getElementById('barChart')?.getContext('2d');

    const barChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [
                'January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December'
            ],
            datasets: [
                {
                    label: 'Income',
                    data: Array(12).fill(0),
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Expenses',
                    data: Array(12).fill(0),
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Monthly Income vs Expenses' }
            },
            scales: { y: { beginAtZero: true } }
        }
    });

    document.getElementById('chart-tab')?.addEventListener('click', () => {
        const monthlyData = getMonthlyData();

        barChart.data.labels = monthlyData.map(({ month }) => 
            month.charAt(0).toUpperCase() + month.slice(1)
        );
        barChart.data.datasets[0].data = monthlyData.map(({ income }) => income);
        barChart.data.datasets[1].data = monthlyData.map(({ expenses }) => expenses);

        barChart.update();
    });

    const downloadBtn = document.getElementById('downloadBtn');
    const canvas = document.getElementById('barChart');

    downloadBtn?.addEventListener('click', () => {
        const image = canvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.href = image;
        link.download = 'chart.png';
        link.click();
    });

    document.getElementById('send-email').addEventListener('click', async () => {
        const emailInput = document.getElementById('email-address');
        const userEmail = emailInput.value;
        const chartCanvas = document.getElementById('barChart');

        // Validate email
        const isValidEmail = (email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
        if (!isValidEmail(userEmail)) {
            alert('Please enter a valid email address.');
            return;
        }

        // Check if the chart exists
        const chartInstance = Chart.getChart(chartCanvas); // Get the Chart.js instance
        if (!chartInstance) {
            alert('Chart not found. Please generate the chart first.');
            return;
        }

        // Generate the chart image as a base64 string
        const chartImage = chartInstance.toBase64Image();

        // Send the email and chart image to the backend
        try {
            const response = await fetch('http://localhost:3000/send-email', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email: userEmail,
                    image: chartImage,
                }),
            });

            if (response.ok) {
                alert('Email sent successfully!');
                emailInput.value = ''; // Clear the email input field
            } else {
                alert('Failed to send email. Please try again.');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while sending the email.');
        }
    });
});
