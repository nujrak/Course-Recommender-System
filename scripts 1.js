document.getElementById('add_education_button').addEventListener('click', function() {
    const educationSection = document.getElementById('education_section');
    const index = educationSection.querySelectorAll('.education_entry').length;

    const educationEntry = document.createElement('div');
    educationEntry.classList.add('education_entry');

    educationEntry.innerHTML = `
        <label for="education_level_${index}">Education Level:</label>
        <select id="education_level_${index}" name="education_level[]">
            <option value="high_school">High School</option>
            <option value="undergraduate">Undergraduate</option>
            <option value="graduate">Graduate</option>
            <option value="postgraduate">Postgraduate</option>
        </select>

        <label for="field_of_study_${index}">Field of Study:</label>
        <select id="field_of_study_${index}" name="field_of_study[]">
            <option value="btech">B.Tech</option>
            <option value="bsc">B.Sc</option>
            <option value="mtech">M.Tech</option>
            <option value="msc">M.Sc</option>
            <option value="mba">MBA</option>
        </select>

        <label for="course_name_${index}">Course Name:</label>
        <select id="course_name_${index}" name="course_name[]">
            <option value="python_programming">Python Programming</option>
            <option value="data_science">Data Science</option>
            <option value="web_development">Web Development</option>
            <option value="machine_learning">Machine Learning</option>
            <option value="digital_marketing">Digital Marketing</option>
        </select>
    `;

    educationSection.appendChild(educationEntry);
});

document.getElementById('userForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const userData = {
        name: document.getElementById('name').value,
        email: document.getElementById('email').value,
        age: document.getElementById('age').value,
        location: document.getElementById('location').value,
        education: Array.from(document.querySelectorAll('.education_entry')).map(entry => ({
            education_level: entry.querySelector('[name="education_level[]"]').value,
            field_of_study: entry.querySelector('[name="field_of_study[]"]').value,
            course_name: entry.querySelector('[name="course_name[]"]').value
        })),
        learning_categories: document.getElementById('learning_categories').value,
        difficulty_level: document.getElementById('difficulty_level').value,
        course_format: document.getElementById('course_format').value,
        learning_goals: document.getElementById('learning_goals').value
    };

    console.log('User Data:', userData);

    // Here you can add the code to send this data to your backend
    // using an API endpoint, e.g., using fetch or axios
    // fetch('/api/user', {
    //     method: 'POST',
    //     headers: {
    //         'Content-Type': 'application/json'
    //     },
    //     body: JSON.stringify(userData)
    // })
    // .then(response => response.json())
    // .then(data => console.log('Success:', data))
    // .catch((error) => console.error('Error:', error));
});
