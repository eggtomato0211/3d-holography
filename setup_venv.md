# 仮想環境のセットアップ手順

## 1. 仮想環境の作成

```powershell
# プロジェクトディレクトリに移動
cd c:\Users\Owner\mizusaki\3d-holography

# 仮想環境を作成
python -m venv venv
```

## 2. 仮想環境の有効化

```powershell
# PowerShellの場合
.\venv\Scripts\Activate.ps1

# コマンドプロンプトの場合
.\venv\Scripts\activate.bat
```

> **注意**: PowerShell でスクリプト実行が許可されていない場合、以下を実行:
>
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

## 3. 依存関係のインストール

```powershell
# pipをアップグレード
python -m pip install --upgrade pip

# requirements.txtから依存関係をインストール
pip install -r requirements.txt
```

## 4. 可視化スクリプトの実行

```powershell
# Plotlyを使った3D可視化
python app\python\3d-imaging\visualize_3d_plotly.py
```

## 5. 仮想環境の無効化

```powershell
deactivate
```

## トラブルシューティング

### PowerShell で Activate.ps1 が実行できない場合

```powershell
# 実行ポリシーを確認
Get-ExecutionPolicy

# RemoteSignedに変更
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### pip の SSL 証明書エラーが出る場合

```powershell
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```
