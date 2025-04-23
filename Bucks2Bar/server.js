const express = require('express');
const nodemailer = require('nodemailer');
const bodyParser = require('body-parser');
const cors = require('cors'); // Import the CORS middleware
require('dotenv').config(); // Load environment variables from .env

const app = express();

// Enable CORS for all origins
app.use(cors());
app.use(bodyParser.json());

// Configure your SMTP server
const transporter = nodemailer.createTransport({
    host: 'smtp.resend.com', // Resend SMTP server
    port: 587, // Resend SMTP port
    auth: {
        user: process.env.RESEND_SMTP_USER, // Load from .env
        pass: process.env.RESEND_SMTP_PASS, // Load from .env
    },
});

app.post('/send-email', async (req, res) => {
    console.log('Send Email works!');
    const { email, image } = req.body;

    try {
        await transporter.sendMail({
            from: 'test@resend.dev', // Use a test/dev email from Resend
            to: email,
            subject: 'Your Chart Image',
            html: '<p>Please find your chart image attached.</p>',
            attachments: [
            {
                filename: 'chart.png',
                content: image.split('base64,')[1],
                encoding: 'base64',
            },
            ],
        });

        res.status(200).send('Email sent successfully');
    } catch (error) {
        console.error('Error sending email:', error);
        res.status(500).send('Failed to send email');
    }
});

app.listen(3000, () => {
    console.log('Server is running on port 3000');
});