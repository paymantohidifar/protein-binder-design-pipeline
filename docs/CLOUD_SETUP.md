# Provisioning a GPU Instance (Lambda Cloud)

Getting GPU access from the large enterprise clouds (Google Cloud, AWS) often means quota requests and high prices. Cheaper options like Vultr exist, but availability can be unpredictable.

For this pipeline, **Lambda Cloud** is a reliable choice: instances are usually available on demand, they come with GPU drivers pre-installed, and the pricing is easy to reason about.

> **Pricing note:** GPU instances bill by the second while running. A persistent filesystem bills continuously for as long as it exists — even when no instance is attached to it.

---

## 1. Create an Account

1. Sign up at [lambda.ai](https://lambda.ai) and verify your email address.
2. Add a payment method under billing settings.
3. Check the current on-demand hourly rates on the dashboard. An **A100 (40 GB)** usually costs about 1.5–2× an **A10 (24 GB)**, but gives you noticeably faster inference.

---

## 2. Add an SSH Key

Lambda instances authenticate with SSH key pairs — password login is disabled by default.

1. Generate a key pair on your **local machine**:
   ```bash
   ssh-keygen -t ed25519 -C "binder-pipeline" -f ~/.ssh/lambda_binder
   ```
   Press `Enter` twice to leave the passphrase empty, or set one if you prefer.

2. Display and copy the public key:
   ```bash
   cat ~/.ssh/lambda_binder.pub
   ```

3. In the Lambda dashboard, go to **SSH Keys → Add SSH Key**, paste the public key, and give it a memorable name.

---

## 3. Create a Persistent Filesystem (Recommended)

> **Important:** create the filesystem **before** launching an instance. A filesystem can only be attached at launch time, and it must be in the same region as the instance.

1. In the dashboard, go to **Filesystems → Create Filesystem**.
2. Pick your target region (California, Virginia, and so on) and a size: **~150 GB** is comfortable for prototyping, **~500 GB** if you plan production-scale campaigns.
3. Give it a name using letters, numbers, and dashes — no spaces or underscores. You will use this name as a directory, so keep it short.

### Why This Matters

Building the pipeline downloads about **50 GB** of container images and **18 GB** of model weights. Without persistent storage you would rebuild the images and re-download the weights on every fresh instance, burning an hour or more of billed GPU time each cycle.

With a filesystem attached, your repository, cached weights, and campaign results all survive instance teardown.

---

## 4. Choose an Instance Type and Image

Go to **Instances → Launch Instance**. Two single-GPU configurations suit this pipeline well:

| Instance Type | VRAM | Local NVMe | vCPUs | RAM | Best For |
| --- | --- | --- | --- | --- | --- |
| **1× A10** | 24 GB PCIe | 1.4 TiB | 30 | 200 GiB | Prototyping and notebook smoke tests (5–10 binders in 30–60 min) |
| **1× A100** | 40 GB SXM4 | 0.5 TiB | 30 | 200 GiB | High-throughput campaigns (roughly 3–4× faster inference) |

Both are supported architectures for this pipeline. Avoid H100 and B200 instances — some of the pinned model binaries cannot run on them.

### Picking a Base Image

Lambda offers three Ubuntu flavors, on both 22.04 LTS and 24.04 LTS:

* **Lambda Stack (recommended):** NVIDIA drivers, the Container Toolkit, Jupyter, and a full host AI stack. This pipeline doesn't use that host stack — every model runs in its own container — but the drivers and Jupyter make interactive prototyping painless.
* **GPU Base:** just the NVIDIA drivers and Container Toolkit. A good fit for headless production runs.
* **Ubuntu Minimal:** no GPU drivers. Not recommended.

For more detail, see the [Lambda on-demand cloud docs](https://docs.lambda.ai/public-cloud/on-demand/).

---

## 5. Launch and Connect

1. Click **Launch Instance**.
2. Select your **instance type**, **region**, and the **filesystem** you created.
3. Once the status turns *Running* (usually 1–2 minutes), copy the instance's public IP address.
4. Connect over SSH from your local terminal:
   ```bash
   ssh -i ~/.ssh/lambda_binder ubuntu@<INSTANCE_IP_ADDRESS>
   ```

> **A note on users:** the default login is **`ubuntu`**, not `root`. Pipeline containers run under your host user ID, so output files are never left locked under root ownership.

---

## 6. Set Up and Verify

Your attached filesystem is mounted inside your home directory, at the mount point shown on the dashboard — typically `/home/ubuntu/<FILESYSTEM_NAME>`. **Anything written outside that directory is lost when the instance is terminated**, so the repository and the weight cache both need to live inside it.

```bash
# Move onto the persistent filesystem (substitute your filesystem's name)
cd /home/ubuntu/<FILESYSTEM_NAME>

# Clone the repository onto persistent storage
git clone https://github.com/paymantohidifar/protein-binder-design-pipeline.git --branch main binder_repo
cd binder_repo

# Point the model weight cache at persistent storage, too
export BINDER_CACHE=/home/ubuntu/<FILESYSTEM_NAME>/binder-cache
echo "export BINDER_CACHE=$BINDER_CACHE" >> ~/.bashrc

# Set up Docker privileges, then run the pre-flight check
bash scripts/setup_docker.sh
bash scripts/bootstrap_instance.sh --check
```

> **Why `BINDER_CACHE`:** the weight cache defaults to `~/.cache/binder-pipeline`, which sits on the instance's boot volume and disappears on teardown. Exporting `BINDER_CACHE` moves those 18 GB onto the filesystem so the next instance starts warm. The same variable is read by `config/campaign.yaml`, so the pipeline and the bootstrap script stay in agreement.

With the pre-flight check passing, continue from **[Step 5 of the README](../README.md#5-build-the-environment-and-cache-weights)** to build the images and cache the weights. Steps 2–4 there are the ones you just completed.

---

## Tearing Down and Managing Costs

* **Terminate the instance promptly.** Delete the GPU instance from the dashboard as soon as a campaign finishes — that stops the hourly charge.
* **Keep the filesystem.** It holds your repository, cached weights, and campaign outputs. Attach it to your next instance and you resume with a warm cache.
* **Remember storage bills separately.** A persistent filesystem costs a small amount per month for as long as it exists. Delete it only when you're finished with the project and have backed up your results.
