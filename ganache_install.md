To download and install Ganache on a Linux server without using `apt`, you can use Node.js and `npm` (Node Package Manager). Below are the steps to install Ganache CLI using `npm` without relying on `apt`.

### Step 1: Install Node.js and npm Without `apt`

Since you cannot use `apt`, you can install Node.js and `npm` using **Node Version Manager (nvm)**. Here's how:

#### a. Install nvm

Run the following command to download and install nvm:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.4/install.sh | bash
```

**Note:** Replace `v0.39.4` with the latest version of nvm if it's updated.

#### b. Load nvm into Your Shell Session

After installation, you need to load nvm. You can do this by running:

```bash
export NVM_DIR="$HOME/.nvm"
source "$NVM_DIR/nvm.sh"
```

Alternatively, you can close and reopen your terminal or source your profile:

```bash
source ~/.bashrc  # or ~/.zshrc if you're using Zsh
```

#### c. Install Node.js Using nvm

Now, install the latest LTS (Long Term Support) version of Node.js:

```bash
nvm install --lts
```

Verify the installation:

```bash
node -v
npm -v
```

### Step 2: Install Ganache Using npm

With Node.js and `npm` installed, you can now install Ganache CLI globally:

```bash
npm install -g ganache
```

### Step 3: Verify Ganache Installation

Check if Ganache is installed correctly by running:

```bash
ganache --version
```
