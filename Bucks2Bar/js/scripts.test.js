describe('Username regex validation', () => {
    const regex = /^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&#])[A-Za-z\d@$!%*?&#]{8,}$/;

    test('Valid username with all criteria met', () => {
        const validUsernames = [
            "Valid@123",
            "TestUser1!",
            "Strong#Password2",
            "Example$456"
        ];

        validUsernames.forEach(username => {
            expect(regex.test(username)).toBe(true);
        });
    });

    test('Invalid username missing capital letter', () => {
        const invalidUsernames = [
            "invalid@123",
            "testuser1!",
            "weak#password2",
            "example$456"
        ];

        invalidUsernames.forEach(username => {
            expect(regex.test(username)).toBe(false);
        });
    });

    test('Invalid username missing special character', () => {
        const invalidUsernames = [
            "Invalid123",
            "TestUser1",
            "StrongPassword2",
            "Example456"
        ];

        invalidUsernames.forEach(username => {
            expect(regex.test(username)).toBe(false);
        });
    });

    test('Invalid username missing number', () => {
        const invalidUsernames = [
            "Invalid@!",
            "TestUser!",
            "Strong#Password",
            "Example$"
        ];

        invalidUsernames.forEach(username => {
            expect(regex.test(username)).toBe(false);
        });
    });

    test('Invalid username shorter than 8 characters', () => {
        const invalidUsernames = [
            "Val@1",
            "T1!",
            "Str#2",
            "Ex$4"
        ];

        invalidUsernames.forEach(username => {
            expect(regex.test(username)).toBe(false);
        });
    });
});