document.getElementById('submit').addEventListener('click', async () => {
    const inputData = document.getElementById('input-data').value;
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({input_data: inputData}),
    });

    const prediction = await response.json();
    document.getElementById('result').textContent = JSON.stringify(prediction);
});
