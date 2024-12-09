document.getElementById('uploadForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    console.log('Uploading image...');
    const formData = new FormData();
    const fileInput = document.getElementById('file');
    const textQuery = document.getElementById('query').value;
    const lamInput = document.getElementById('lam');
    const dropdownSelection = document.getElementById('dropdown').value;

    formData.append('file', fileInput.files[0]);
    formData.append('query', textQuery);
    formData.append('lam', lamInput.value);
    formData.append('dropdown', dropdownSelection);

    const response = await fetch('/compress', {
        method: 'POST',
        body: formData,
    });

    const imageBlob = await response.blob();
    // const imageURL = URL.createObjectURL(imageBlob);

    const compressedImage = document.getElementById('compressedImage');
    compressedImage.src = imageURL;
    compressedImage.style.display = 'block';
});
