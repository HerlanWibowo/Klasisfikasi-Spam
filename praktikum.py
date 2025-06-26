# spam_app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#------------------------------------------------#
# Fungsi untuk Melatih Model (dengan caching)
# Kode di dalam fungsi ini tidak berubah
#------------------------------------------------#
@st.cache_data
def train_model():
    """
    Fungsi ini akan melatih model Naive Bayes dan mengembalikan
    objek model, vectorizer, akurasi, dan dataframe yang digunakan.
    """
    # Dataset 100 data (50 'bukan_spam', 50 'spam')
    data = {
        'teks': [
            # =============== 50 Contoh BUKAN SPAM ===============
            'Hai, apa kabar?',
            'Besok ada rapat penting jam 9 pagi di kantor.',
            'Tolong kirim laporannya sebelum makan siang ya, terima kasih.',
            'Jadwal ujian semester ini sudah bisa diakses di portal akademik.',
            'Jangan lupa bayar tagihan listrik bulan ini sebelum tanggal 20.',
            'Nanti kita makan malam di warung sate langganan saja.',
            'Paket Anda dengan nomor resi JD12345 telah tiba di gudang sortir Jakarta.',
            'Terima kasih banyak atas konfirmasi dan kerja samanya.',
            'Aku lagi di jalan, mungkin telat sekitar 10 menit karena macet.',
            'Meeting hari ini diundur jadi jam 2 siang karena ada kendala teknis.',
            'Sisa saldo pulsa Anda akan segera habis, segera lakukan isi ulang.',
            'Kode OTP Anda adalah 8821. JANGAN BERIKAN PADA SIAPAPUN.',
            'Oke, nanti aku kabari lagi secepatnya setelah dapat info.',
            'Informasi mengenai pendaftaran ulang mahasiswa baru bisa dilihat di website resmi universitas.',
            'Apakah produk yang kemarin saya tanyakan di toko masih tersedia?',
            'Selamat ulang tahun! Semoga panjang umur dan sehat selalu.',
            'Mohon review draf proposal yang sudah saya kirimkan via email.',
            'Jadwal penerbangan Anda adalah besok jam 7 pagi, jangan sampai terlambat.',
            'Aku sudah sampai di depan rumahmu.',
            'Nilai tugas besar sudah keluar, silakan cek.',
            'Ada yang punya catatan pertemuan minggu lalu?',
            'Resep rendang yang kamu kasih kemarin enak banget!',
            'Maaf, nomor yang Anda tuju sedang tidak dapat dihubungi.',
            'Cuaca hari ini sepertinya akan hujan, jangan lupa bawa payung.',
            'Film yang kita tonton semalam sangat bagus, ceritanya tidak terduga.',
            'Saya setuju dengan poin-poin yang disampaikan dalam presentasi.',
            'Pembayaran tagihan kartu kredit Anda untuk bulan Juni telah berhasil.',
            'Lokasi acara berada di ballroom hotel, lantai 3.',
            'Jangan lupa bawa laptop untuk praktikum besok.',
            'Tugas kelompok kita dikumpulkan paling lambat hari Jumat.',
            'Kucingku tadi pagi melahirkan 3 anak, lucu sekali.',
            'Hati-hati di jalan ya, kabari kalau sudah sampai.',
            'Ada update baru untuk aplikasi WhatsApp, silakan perbarui.',
            'Saya akan mengambil cuti pada tanggal 28-30 Juni.',
            'Pekerjaan perbaikan jalan di area ini diperkirakan selesai minggu depan.',
            'Antrian di bank hari ini sangat panjang.',
            'Bisa tolong bantu saya angkat galon ini?',
            'Jadwal kereta api ke Bandung tersedia pada jam 8 dan 10 pagi.',
            'Hasil tes lab Anda sudah keluar dan semuanya normal.',
            'Mohon untuk tidak parkir di depan gerbang.',
            'Laporan penjualan kuartal ini menunjukkan peningkatan sebesar 15%.',
            'Acara akan dimulai 15 menit lagi.',
            'Kalau ada waktu luang, ayo kita main bulu tangkis.',
            'Gojek: Driver Anda akan segera tiba.',
            'Terima kasih telah berbelanja di toko kami.',
            'Bisakah kita menjadwalkan ulang pertemuan kita?',
            'Saya lupa kata sandi akun saya, bagaimana cara meresetnya?',
            'Selamat hari raya Idul Fitri, mohon maaf lahir dan batin.',
            'Dokumen yang Anda minta sudah saya lampirkan di email.',
            'Rapat orang tua murid akan diadakan hari Sabtu.',

            # =============== 50 Contoh SPAM ===============
            'Dapatkan diskon spesial 50% khusus untuk Anda HARI INI!',
            'MENANGKAN VOUCHER JUTAAN RUPIAH, KLAIM SEKARANG JUGA!',
            'Promo terbatas! Beli 1 gratis 1, klik link di bio IG kami',
            'Selamat! Nomor Anda terpilih sebagai pemenang undian senilai 100 juta dari Bank ABC.',
            'Pinjaman online cepat cair dalam 5 menit tanpa jaminan, daftar sekarang juga!',
            'Butuh dana cepat? Kami solusinya! Hubungi nomor di bawah ini untuk info lebih lanjut.',
            'RAHASIA KURUS DALAM 7 HARI TANPA OLAHRAGA! Klik link ini untuk membuktikannya!',
            'Lowongan kerja dari rumah, gaji puluhan juta per bulan. Minat? Hubungi WA ini.',
            'Hanya dengan KTP, dapatkan limit pinjaman hingga 20 juta rupiah tanpa verifikasi.',
            'Tingkatkan traffic website Anda dengan layanan SEO terbaik dan termurah dari kami.',
            'Perbesar alat vital Anda dengan ramuan herbal dari pedalaman. Garansi uang kembali!',
            'Daftar gratis di situs kami dan dapatkan bonus saldo 100rb untuk member baru, terbatas!',
            'Main game dapat uang? Download aplikasi kami sekarang dan buktikan sendiri hasilnya!',
            'Asuransi jiwa dengan premi ringan, lindungi keluarga Anda dari sekarang. Hubungi agen kami.',
            'Satu-satunya cara menjadi kaya raya tanpa bekerja keras, hubungi kami segera!',
            'SELAMAT ANDA MENDAPATKAN 1 UNIT MOBIL! Untuk info klaim hubungi CS kami.',
            'Akun DANA Anda mendapatkan cashback Rp 500.000, segera cek aplikasinya.',
            'Investasi saham dengan keuntungan pasti 30% per bulan. Anti rugi!',
            'Putihkan kulit Anda secepat kilat dengan serum viral dari Korea. Order sekarang!',
            'Nonton video-video viral terbaru tanpa sensor, klik di sini.',
            'Jasa followers Instagram permanen dan murah, proses cepat.',
            'Solusi kebotakan, rambut tumbuh lebat dalam 30 hari. Terbukti!',
            'Nomor togel jitu, dijamin tembus 4 angka malam ini. Chat untuk maharnya.',
            'Temukan pasangan panas di dekatmu, daftar gratis di aplikasi kami.',
            'Dibutuhkan admin online, kerja 2 jam/hari gaji 5 juta. Tidak perlu pengalaman.',
            'Hapus data pinjol ilegal Anda secara permanen. Kami bisa bantu.',
            'Obat kuat herbal, bikin istri makin sayang. Pesan sekarang!',
            'Anda tidak akan percaya apa yang terjadi pada artis ini. Cek fotonya!',
            'Ingin tahu siapa yang sering stalking profil media sosial Anda? Cek dengan tool ini.',
            'Tawaran eksklusif hanya untuk Anda: iPhone 15 Pro Max hanya 5 juta.',
            'Modal receh untung jutaan, gabung di grup trading kami.',
            'Sertifikat vaksin ke-3 tanpa suntik, proses cepat dan resmi.',
            'Cara mendapatkan kuota internet gratis 100GB untuk semua operator.',
            'Cek! Mungkin nama Anda ada di daftar penerima bantuan sosial 2 juta rupiah.',
            'Jasa hacking akun sosial media, dijamin aman dan rahasia.',
            'Kredit HP tanpa DP, cicilan 0% hanya di toko kami.',
            'Rahasia awet muda para artis akhirnya terungkap! Baca selengkapnya.',
            'Software penambah saldo e-wallet otomatis. Download gratis.',
            'Undangan pernikahan digital, dapatkan diskon 70% untuk pemesanan hari ini.',
            'Selamat! Anda memenangkan giveaway dari influencer favorit Anda.',
            'Tingkatkan peluang menang slot online dengan aplikasi injector kami.',
            'Butuh uang mendesak? Gadai BPKB motor, langsung cair.',
            'Miliki penghasilan pasif hingga puluhan juta dari internet. Pelajari caranya.',
            'Jangan biarkan data pribadi Anda tersebar! Gunakan layanan VPN kami.',
            'SELAMAT! Profil Anda terpilih untuk mendapatkan hadiah spesial.',
            'Kerja mudah hanya like dan subscribe video YouTube, dapatkan komisi harian.',
            'Saldo ShopeePay gratis untuk 20 orang pertama yang membalas pesan ini.',
            'Cukup KTP dan KK, dana cair 10 juta dalam hitungan menit.',
            'Berita mengejutkan! Skandal besar terungkap, baca selengkapnya di sini.',
            'Jadilah jutawan berikutnya dengan bergabung di bisnis networking kami.'
        ],
        'label': (['bukan_spam'] * 50) + (['spam'] * 50)
    }
    df = pd.DataFrame(data)

    X = df['teks']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    vectorizer = CountVectorizer(stop_words=['di', 'dan', 'yang', 'untuk', 'ini', 'itu', 'dengan', 'ke', 'dari'])
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    return model, vectorizer, accuracy, df

#------------------------------------------------#
# Layout Aplikasi Streamlit
#------------------------------------------------#

# Muat model, vectorizer, dan data yang telah dilatih
model, vectorizer, accuracy, df = train_model()

# Judul Aplikasi
st.title("🤖 Aplikasi Klasifikasi Spam")
st.markdown("Aplikasi ini menggunakan Algoritma **Naive Bayes** untuk memprediksi apakah sebuah teks adalah spam atau bukan.")
st.write("---")

# --- BAGIAN PENJELASAN MODEL ---
st.header("Bagaimana Cara Kerja Model Ini?")
st.markdown("""
Aplikasi ini menggunakan model *machine learning* bernama **Multinomial Naive Bayes**. Cara kerjanya bisa kita bayangkan seperti memiliki dua ember ajaib:
""")

col1, col2 = st.columns(2)
with col1:
    st.info("**Ember 'BUKAN SPAM'** 🔵")
    st.markdown("""
    - Saat training, model membaca semua contoh pesan yang BUKAN SPAM.
    - Setiap kata dari pesan tersebut (seperti "rapat", "laporan", "jadwal", "besok") dimasukkan ke dalam ember ini.
    """)
with col2:
    st.error("**Ember 'SPAM'** 🔴")
    st.markdown("""
    - Model juga membaca semua contoh pesan SPAM.
    - Setiap kata dari pesan spam (seperti "promo", "gratis", "diskon", "menang") dimasukkan ke dalam ember ini.
    """)

st.markdown("""
> Setelah proses training, kedua ember ini sudah penuh dengan "kamus kata" untuk masing-masing kategori.

**Saat Anda memasukkan teks baru untuk diprediksi:**
1.  Model akan "memecah" teks Anda menjadi kata-kata.
2.  Untuk setiap kata, model akan memeriksa: *"Seberapa sering kata ini muncul di Ember SPAM dibandingkan di Ember BUKAN SPAM?"*
3.  Model kemudian menghitung skor probabilitas total. Jika skor untuk "SPAM" lebih tinggi, maka teks Anda akan diklasifikasikan sebagai SPAM, dan sebaliknya.

**Mengapa disebut 'Naive' (Naif)?**
Model ini secara *naif* mengasumsikan bahwa setiap kata dalam kalimat tidak saling berhubungan. Ia tidak peduli jika kata `kartu` muncul setelah kata `kredit`. Ia hanya melihat probabilitas kemunculan `kartu` dan `kredit` secara terpisah. Meskipun asumsi ini salah dalam tata bahasa nyata, pendekatan ini terbukti sangat cepat dan efektif untuk klasifikasi teks.
""")
st.write("---")
# --- AKHIR BAGIAN PENJELASAN ---


# Form Input dari Pengguna
st.header("Masukkan Teks untuk Diperiksa")
with st.form(key='text_form'):
    user_input = st.text_area(
        "Ketik atau tempelkan pesan Anda di sini:",
        "Selamat! Anda mendapatkan cashback 500rb dari transaksi terakhir Anda. Segera klaim di link berikut!",
        height=150
    )
    submit_button = st.form_submit_button(label='Klasifikasikan!')

# Proses ketika tombol ditekan
if submit_button:
    if user_input:
        # Lakukan prediksi dan tampilkan hasil
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)
        prediction_proba = model.predict_proba(input_vec)

        st.subheader("Hasil Prediksi")
        if prediction[0] == 'spam':
            st.error("🚨 Pesan ini terindikasi sebagai: SPAM")
        else:
            st.success("✅ Pesan ini terindikasi sebagai: BUKAN SPAM")

        st.write("---")
        st.write("**Probabilitas:**")
        col1_res, col2_res = st.columns(2)
        with col1_res:
            st.metric(label=f"Probabilitas '{model.classes_[0]}'", value=f"{prediction_proba[0][0]:.2%}")
        with col2_res:
            st.metric(label=f"Probabilitas '{model.classes_[1]}'", value=f"{prediction_proba[0][1]:.2%}")
    else:
        st.warning("Mohon masukkan teks terlebih dahulu untuk diklasifikasikan.")

# Menampilkan detail model dan data training di bagian bawah
st.write("---")
with st.expander("Lihat Detail Model Teknis dan 100 Data Latih"):
    st.write("Model ini dilatih menggunakan dataset yang lebih besar dan beragam:")
    st.dataframe(df)
    st.write(f"**Algoritma:** Multinomial Naive Bayes")
    st.write(f"**Jumlah Data Latih:** {len(df)} entri")
    st.write(f"**Akurasi Model pada Data Uji Internal:** **{accuracy:.2%}**")