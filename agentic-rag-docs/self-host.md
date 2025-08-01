# Using Your Own GPU for Inference

**Who is this guide for?** 
- People that want to run containerized inference on a GPU
- People that have some experience with using containers
- People that have experience with configuring and using remote resources

**What are the guide limitations?** 
- It assumes you have the remote already setup with appropriate dependencies, i.e. NVIDIA GPU drivers, the Container Toolkit, 
  and either Docker or Podman
- It assumes that Ubuntu 22.04 or Ubuntu 24.04 LTS is the reference OS
- Some steps may need adjustment 

**What else do I need to know?**
- You will need root/sudo access for most setup steps
- The first place to check for issues is `nvidia-smi` output
- Make sure your GPU meets the minimum requirements before starting

For detailed software installation instructions, see:
- [NVIDIA Driver Installation](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
- [NVIDIA Container Toolkit Setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Docker Installation](https://docs.docker.com/engine/install/ubuntu/) or [Podman Installation](https://podman.io/docs/installation)

---

## High-Level Overview

There are many ways you can setup inference on a remote GPU. 

This guide focuses on running a lightweight Ollama container.

### Ollama Container
 - Easier to set up and manage
 - Runs many models out of the box
 - Good for experimentation and development
 - See [Ollama Setup Guide](#ollama-setup) below
 - GPU requirements vary by model but can be as low as 8 GB of vRAM
 - Version 0.10.0 or newer is recommended to enable advanced sampling options

## General Prerequisites

- Make sure the remote is properly setup and that you have SSH access to it
  - IP address: ``<remote-ip>``
  - Remote user: ``<remote-user>``
- Be in a terminal session on the remote
- Make sure it's open to TCP access on a known port, i.e. ``<remote-port>``
- Make sure that the container runtime is properly configured

# Option A: Using Ollama

### Prerequisites for Ollama
- Satisfy the [General Prerequisites](#general-prerequisites)

### Three Basic Steps
- **Deploy Ollama Container**: Pull the Ollama container onto the remote and run it
- **Model Pulled Automatically**: The container uses the `OLLAMA_MODEL` environment variable to download a model at startup
- **Add Ollama Container as an Endpoint**: Configure the Agentic RAG app to use the model

## Deploy Ollama Container

Do the following in the **remote** terminal. If you encounter any errors, use an LLM to help you debug. 

```bash
# Pull the Ollama container. Change the tag if you want a different one. 

docker pull ollama/ollama:0.10.0

# Run it and make sure to connect the port on the remote, <remote-port>, to the Ollama port in the container, 11434

docker run -d --name ollama \
  --gpus all --restart unless-stopped \
  -p <remote-port>:11434 \
  -v ollama_data:/root/.ollama \
  -e OLLAMA_MODEL=llama3-chatqa:8b \
  ollama/ollama:0.10.0

# Make sure it is running properly

curl http://localhost:10000/api/tags

```

The container automatically pulls the model specified by the `OLLAMA_MODEL` environment variable when it starts.
Verify the model has been pulled and is available:

```bash
curl http://localhost:<remote-port>/api/tags
```

## Add Ollama Container as an Endpoint

Do the following in a **local** terminal. If you encounter any errors, use an LLM to help you debug. 

You will need the remote ip, ``<remote-ip>``, and the remote port, ``<remote-port>``.

```bash

# Test local access to the Ollama container on the remote 

curl -X POST http://<remote-ip>:<remote-port>/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3-chatqa:8b", "prompt": "Hello, Ollama!"}'
```



1. Open the **Agentic RAG** project in AI Workbench
2. Select **Models**
3. For each component you want to use the Ollama container:
   1. Select **Self-Hosted Endpoint**
   2. Enter the ip address, port and model name



